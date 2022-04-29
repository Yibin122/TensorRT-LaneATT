#include "ganet.hh"
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("../02610.jpg");
    GANet model("../ganet.trt8");
    model.Detect(img);
    return 0;
}
