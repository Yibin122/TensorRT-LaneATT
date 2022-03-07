#include "laneatt.hh"

#include <fstream>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define INPUT_H 360
#define INPUT_W 640
#define N_OFFSETS 72
#define N_STRIPS (N_OFFSETS - 1)
#define MAX_COL_BLOCKS 1000

LaneATT::LaneATT(const std::string& plan_path) {
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = MAX_COL_BLOCKS * (5 + N_OFFSETS);

    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    image_data_.resize(buffer_size_[0]);
    detections_.resize(MAX_COL_BLOCKS);

    cudaStreamCreate(&stream_);
    LoadEngine(plan_path);
}

LaneATT::~LaneATT() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
}

void LaneATT::DetectLane(const cv::Mat& raw_image) {
    // Preprocessing
    cv::Mat img_resize;
    cv::resize(raw_image, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    // img_resize.convertTo(img_resize, CV_32FC3, 1.0);
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = data_hwc[j * 3 + c] / 255.f;
        }
    }

    // Do inference
    cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->execute(1, &buffers_[0]);
    cudaMemcpyAsync(detections_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    // NMS and decoding
    PostProcess(img_resize);
}

void LaneATT::LoadEngine(const std::string& engine_file) {
    std::ifstream in_file(engine_file, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_file << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    // FIXME: getPluginCreator could not find plugin: ScatterND version: 1
    initLibNvInferPlugins(&g_logger_, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    runtime->destroy();
}

float LaneIoU(const Detection& a, const Detection& b) {
    int start_a = static_cast<int>(a.start_y * N_STRIPS + 0.5f);
    int start_b = static_cast<int>(b.start_y * N_STRIPS + 0.5f);
    int start = std::max(start_a, start_b);
    int end_a = start_a + static_cast<int>(a.length + 0.5f) - 1;
    int end_b = start_b + static_cast<int>(b.length + 0.5f) - 1;
    int end = std::min(std::min(end_a, end_b), N_STRIPS);
    // if (end < start) {
    //     return 1.0f / 0.0f;
    // }
    float dist = 0.0f;
    for (int i = start; i <= end; ++i) {
        dist += fabs(a.lane_xs[i] - b.lane_xs[i]);
    }
    dist /= static_cast<float>(end - start + 1);
    return dist;
}

void LaneATT::PostProcess(cv::Mat& lane_image, float conf_thresh, float nms_thresh, int nms_topk) {
    // 1.Do NMS
    std::vector<Detection> candidates;
    std::vector<Detection> proposals;
    for (auto det : detections_) {
        if (det.score > conf_thresh) {
            candidates.push_back(det);
        }
    }
    // std::cout << candidates.size() << std::endl;
    std::sort(candidates.begin(), candidates.end(), [=](const Detection& a, const Detection& b) { return a.score > b.score; });
    for (int i = 0; i < candidates.size(); ++i) {
        if (candidates[i].score < 0.0f) {
            continue;
        }
        proposals.push_back(candidates[i]);
        if (proposals.size() == nms_topk) {
            break;
        }
        for (int j = i + 1; j < candidates.size(); ++j) {
            if (candidates[j].score > 0.0f && LaneIoU(candidates[j], candidates[i]) < nms_thresh) {
                candidates[j].score = -1.0f;
            }
        }
    }

    // 2.Decoding
    std::vector<float> anchor_ys;
    for (int i = 0; i < N_OFFSETS; ++i) {
        anchor_ys.push_back(1.0f - i / float(N_STRIPS));
    }
    std::vector<std::vector<cv::Point2f>> lanes;
    for (const auto& lane: proposals) { 
        int start = static_cast<int>(lane.start_y * N_STRIPS + 0.5f);
        int end = start + static_cast<int>(lane.length + 0.5f) - 1;
        end = std::min(end, N_STRIPS);
        std::vector<cv::Point2f> points;
        for (int i = start; i <= end; ++i) {
            points.push_back(cv::Point2f(lane.lane_xs[i], anchor_ys[i] * INPUT_H));
        }
        lanes.push_back(points);
    }

    // 3.Visualize
    for (const auto& lane_points : lanes) {
        for (const auto& point : lane_points) {
            cv::circle(lane_image, point, 1, cv::Scalar(0, 255, 0), -1);
        }
    }
    cv::imshow("laneatt_trt", lane_image);
    cv::waitKey(0);
}
