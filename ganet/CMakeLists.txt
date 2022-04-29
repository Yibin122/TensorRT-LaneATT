cmake_minimum_required(VERSION 2.8.0)
project(ganet_trt)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER NAMES nvinfer)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ~/TensorRT-8.0.3.4/include
)

add_executable(${PROJECT_NAME}
  ganet.cc
  main.cc
)

target_link_libraries(${PROJECT_NAME}
  # ${NVINFER}
  # ${NVINFER_PLUGIN}
  ~/TensorRT-8.0.3.4/lib/libnvinfer.so
  ~/TensorRT-8.0.3.4/lib/libnvinfer_plugin.so
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
