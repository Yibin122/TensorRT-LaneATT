#include "laneatt.hh"
#include <opencv2/imgcodecs.hpp>

int main() {
    cv::Mat img = cv::imread("../02610.jpg");
    LaneATT model("../LaneATT.trt8");
    model.DetectLane(img);
    return 0;
}
