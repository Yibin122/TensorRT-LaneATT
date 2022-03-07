cmake_minimum_required(VERSION 2.8.0)
project(laneatt_test)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
# /usr/lib/x86_64-linux-gnu/libnvinfer.so
# /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
find_library(NVINFER NAMES nvinfer)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)

include_directories(
  ${CUDA_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
  laneatt_test.cc
  laneatt.cc
)

target_link_libraries(${PROJECT_NAME}
  ${NVINFER}
  ${NVINFER_PLUGIN}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
