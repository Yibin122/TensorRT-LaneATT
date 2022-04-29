# TensorRT-GANet
This repo deploys [GANet](https://github.com/Wolfwjs/GANet) using TensorRT.

## Steps
1. Install [TensorRT 8.0](https://developer.nvidia.cn/nvidia-tensorrt-8x-download)
   ```Shell
   # https://developer.nvidia.cn/nvidia-tensorrt-8x-download
   tar -xvzf TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz
   # export TENSORRT_DIR=~/TensorRT-8.0.3.4
   git clone -b release/8.0 https://github.com/Yibin122/TensorRT.git  # MMCVDeformConv2d plugin
   cd TensorRT/
   git submodule update --init --recursive
   mkdir -p build && cd build
   cmake .. -DTRT_LIB_DIR=~/TensorRT-8.0.3.4/lib -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=10.2
   make -j$(nproc)
   make install
   ```
2. PyTorch to ONNX
   ```bash
   git clone https://github.com/Wolfwjs/GANet.git
   cd GANet
   python setup.py develop
   # 用当前repo的deform_conv.py替换: GANet/mmdet/ops/dcn/deform_conv.py
   python ganet_pth2onnx.py
   ```
3. ONNX to TensorRT
   ```bash
   cd ~/TensorRT-8.0.3.4/bin
   ./trtexec --workspace=4096 --onnx=${model_path}/ganet.onnx --saveEngine=${model_path}/ganet.trt8
   ```

## TODO
- [ ] implement postprocessing
