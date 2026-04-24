#include "include/yolov8_seg_pt.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>


yolov8_seg_pt::yolov8_seg_pt(const std::string& model_path,
                            float conf_threshold,
                            float iou_threshold,
                            int img_size)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolov8_seg_pt"),
      session(nullptr),
      session_options(),
      allocator_()
{
    // Initialize ONNX Runtime session
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = new Ort::Session(env, model_path.c_str(), session_options);
}

//segmentation preprocess
cv::Mat yolov8_seg_pt::preprocess(const cv::Mat& image)
{
    // Implement preprocessing logic to resize, normalize, and convert the input image to the format expected by the model
    // This typically involves resizing the image to img_size x img_size, normalizing pixel values, and converting to a tensor format
    cv::Mat preprocessed_image;
    return preprocessed_image;
}

//segmentation postprocess
std::vector<Segmenation> yolov8_seg_pt::postprocess(const float*  output,
                                       int64_t        num_dets,
                                       int64_t        num_classes,
                                       float          scale,
                                       int            pad_w,
                                       int            pad_h,
                                       int            orig_w,
                                       int            orig_h) const
{
    std::vector<Segmenation> segmentations;
    // Implement post-processing logic to convert model output to segmentations
    // This typically involves applying confidence thresholding, non-maximum suppression, and resizing masks to original image dimensions
    return segmentations;
}

//draw segmentation masks and bounding boxes on the original image (testing purpose)