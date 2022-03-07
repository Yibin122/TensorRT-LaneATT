#pragma once

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>

class Logger : public nvinfer1::ILogger {
  public:
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportable_severity(severity) {}

    void log(Severity severity, const char* msg) noexcept {
        if (severity > reportable_severity) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportable_severity;
};

struct Detection {
    float unknown;
    float score;
    float start_y;
    float start_x;
    float length;
    float lane_xs[72];
};

class LaneATT {
  public:
    LaneATT(const std::string& plan_path);

    ~LaneATT();

    void DetectLane(const cv::Mat& raw_image);

  private:
    void LoadEngine(const std::string& engine_file);

    void PostProcess(cv::Mat& lane_image, float conf_thresh=0.4f, float nms_thresh=50.f, int nms_topk=4);

    Logger g_logger_;
    cudaStream_t stream_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* buffers_[2];
    int buffer_size_[2];
    std::vector<float> image_data_;
    std::vector<Detection> detections_;
};
