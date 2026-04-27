#pragma once

#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct SegDetection {
    int      class_id;
    float    confidence;
    cv::Rect box;       // bounding box in original-image pixel space
    cv::Mat  mask;      // binary mask (CV_8UC1), same size as original image
};

// ---------------------------------------------------------------------------
// YOLOv8-seg detector
// ---------------------------------------------------------------------------
//
//  YOLOv8-seg ONNX has TWO output tensors:
//    output0  [1, 4+num_classes+32, num_anchors]  — boxes + class scores + mask coefficients
//    output1  [1, 32, mask_h, mask_w]             — prototype masks (usually 160x160)
//
class YOLOv8Seg {
public:
    explicit YOLOv8Seg(const std::string& model_path,
                       float conf_thresh = 0.25f,
                       float iou_thresh  = 0.45f,
                       int   input_size  = 640);

    std::vector<SegDetection> detect(const cv::Mat& image);

    cv::Mat draw(const cv::Mat&                    image,
                 const std::vector<SegDetection>&  detections,
                 const std::vector<std::string>&   class_names = {},
                 float                             alpha       = 0.45f) const;

private:
    cv::Mat preprocess(const cv::Mat& image,
                       float& scale, int& pad_w, int& pad_h) const;

    std::vector<SegDetection>
    postprocess(const float* det_data,   int64_t num_anchors, int64_t num_classes,
                const float* proto_data, int64_t nm,
                int proto_h,            int proto_w,
                float scale,            int pad_w, int pad_h,
                int orig_w,             int orig_h) const;

    cv::Mat assembleMask(const float*    proto_data,
                         int64_t         nm,
                         int             proto_h,
                         int             proto_w,
                         const float*    coeffs,
                         const cv::Rect& box,
                         int             orig_w,
                         int             orig_h,
                         float           scale,
                         int             pad_w,
                         int             pad_h) const;

    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    Ort::Session        session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output0_name_;
    std::string output1_name_;

    int   input_size_;
    float conf_thresh_;
    float iou_thresh_;

    static constexpr int kPaletteSize = 20;
    static const cv::Scalar kPalette[kPaletteSize];
};