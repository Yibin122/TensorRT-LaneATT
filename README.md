# TensorRT-LaneATT

This repository creates a TensorRT engine of [LaneATT](https://github.com/lucastabelini/LaneATT) for inference.

## 0.Prerequisites
- TensorRT 8.0.3
- CUDA 10.2
- See [install](https://github.com/lucastabelini/LaneATT#2-install)

## 1.ONNX
```
git clone https://github.com/lucastabelini/LaneATT.git
cd LaneATT
python laneatt_to_onnx.py
```

## 2.TensorRT
```
python onnx_to_tensorrt.py
```
![sample](/samples/02610_pred.png)

## TODO
- [x] C++ inference
- [ ] Measure speed on Xavier
