# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os
import itertools
import time
import numpy as np

from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau

norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)

def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0):
    if not cfg.MODEL.SimrelRCNN.USE_HUNGARIAN_LOSS:
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if slow_heads is not None:
                for item in slow_heads:
                    if item in key:
                        logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                        lr = lr / slow_ratio
                        break
            params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]

        optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        return optimizer
    else:
        params = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if slow_heads is not None:
                for item in slow_heads:
                    if item in key:
                        logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                        lr = lr / slow_ratio
                        break
            params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim
        
        if cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            #optimizer = torch.optim.SGD(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif cfg.SOLVER.OPTIMIZER == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            #optimizer = torch.optim.AdamW(
                params, cfg.SOLVER.BASE_LR
            )
        return optimizer


def make_lr_scheduler(cfg, optimizer, logger=None):
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )
    
    else:
        raise ValueError("Invalid Schedule Type")
