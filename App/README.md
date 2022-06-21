## Keypoint Predictions in C++

### Description:
Load a FasterRCNN keypoint model from PyTorch into C++. Stream webcam with OpenCV and draw determined keypoints into the image. 

### Installation requirements (on Ubuntu 20.04):
* `g++`, `cmake`
* `opencv` 
    * [installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
    * requires: `zlib1g-dev`, `libpng-dev`, `libjpeg-turbo8-dev`
* cuda 11.3 
    * [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) | [installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) | [download](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
    * torchvision doesn't seem to run with cuda 11.7
    * if `nvidia-driver-465` (default driver in cuda 11.3 installation) fails to install: pre-install `nvidia-driver-470` individually and install cuda by [runfile](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile), deactivate driver installation in the process
* cudNN
    * [cudNN compatibility](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) | [installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
    * installation file references cuda 11.6 but is backward compatible to 11.3
* torch
    * [installation](https://pytorch.org/cppdocs/installing.html) | [download](https://pytorch.org/get-started/locally/)
* torchvision
    * for (pretrained) keypoint model
    * [installation](https://github.com/pytorch/vision) see "Using the models on C++"
    * build with `-DWITH_CUDA=on`
    
