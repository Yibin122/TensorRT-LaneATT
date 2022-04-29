#include "ganet.hh"

#include <fstream>
#include <memory>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define INPUT_H 320
#define INPUT_W 800
#define KPT_THRESH 0.4f

GANet::GANet(const std::string& engine_path) {
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = 40 * 100 * 2;
    buffer_size_[2] = 40 * 100 * 2;
    buffer_size_[3] = 40 * 100;
    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    cudaMalloc(&buffers_[3], buffer_size_[3] * sizeof(float));
    image_data_.resize(buffer_size_[0]);
    offset_map_.resize(buffer_size_[1]);
    error_map_.resize(buffer_size_[2]);
    confidence_map_.resize(buffer_size_[3]);
    cudaStreamCreate(&stream_);
    LoadEngine(engine_path);
}

GANet::~GANet() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
}

void GANet::Detect(const cv::Mat& raw_img) {
    // Preprocessing
    cv::Mat img_resize;
    cv::resize(raw_img, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    float mean[3] {75.3f, 76.6f, 77.6f};
    float std[3] = {50.5f, 53.8f, 54.3f};
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = (data_hwc[j * 3 + c] - mean[c]) / std[c];
        }
    }

    // Do inference
    cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->executeV2(&buffers_[0]);
    cudaMemcpyAsync(offset_map_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(error_map_.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // FIXME: MMCVDeformConv2d plugin inconsistency
    cudaMemcpyAsync(confidence_map_.data(), buffers_[3], buffer_size_[3] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // cudaStreamSynchronize(stream_);
}

void GANet::LoadEngine(const std::string& engine_path) {
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    initLibNvInferPlugins(&g_logger_, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    runtime->destroy();
}
