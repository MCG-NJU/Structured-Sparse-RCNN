## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- PyTorch >= 1.2 (Mine 1.4.0 (CUDA 10.1))
- torchvision >= 0.4 (Mine 0.5.0 (CUDA 10.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name scene_graph_benchmark
conda activate scene_graph_benchmark

# this installs the right pip and dependencies for the fresh python
#conda install ipython
#conda install scipy
#conda install h5py
pip install --user ipython
pip install --user scipy
pip install --user h5py
pip install --user pyyaml
pip install --user yacs
pip install --user scipy
pip install --user h5py
pip install --user tqdm
pip install --user opencv-python


conda install ipython 
conda install scipy 
conda install h5py 
conda install pyyaml 
conda install yacs 
conda install tqdm 
#conda install opencv-python 
conda install ninja
conda install cython 
conda install overrides
conda install matplotlib

# scene_graph_benchmark and coco api dependencies
#pip install ninja yacs cython matplotlib tqdm opencv-python overrides
pip install --user ninja yacs cython matplotlib tqdm opencv-python overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
#conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
#pip install --user torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

#pip install --user pycocotools ?


# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

#pip install --user apex ?


# install PyTorch Detection
#cd $INSTALL_DIR
#git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
#cd scene-graph-benchmark
cd /root/lmwang/tengyao/SGGbench

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

pip install --upgrade --user scipy



unset INSTALL_DIR


```