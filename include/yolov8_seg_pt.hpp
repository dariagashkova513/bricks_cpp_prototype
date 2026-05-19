#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

constexpr float CONF_POSTPROCESS = 0.85f;
// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct WBFCluster {
    std::vector<int>   members;   // indices into all_detections
    cv::Rect2f         fused_box;
    float              fused_score;
    int                fused_class;
};

struct SegDetection {
    int      class_id;
    float    confidence;
    cv::Rect box;       // bounding box in original-image pixel space
    cv::Mat  mask;      // binary mask (CV_8UC1), same size as original image
    cv::Point mask_origin;
};

struct Tile {
    cv::Mat image;      // 640x640 canvas (gray-padded at edges)
    int     origin_x;   // top-left x in global image coords
    int     origin_y;   // top-left y in global image coords
    int     valid_w;    // actual content width  (< 640 at right border)
    int     valid_h;    // actual content height (< 640 at bottom border)
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
                       float conf_thresh = 0.5f,
                       float iou_thresh  = 0.15f,
                       int   input_size  = 640);

    std::vector<SegDetection> detect(const cv::Mat& image, const float conf_postprocess = CONF_POSTPROCESS);

    cv::Mat draw(const cv::Mat&                    image,
                 const std::vector<SegDetection>&  detections,
                 const std::vector<std::string>&   class_names = {},
                 float                             alpha       = 0.45f) const;

private:

    std::vector<Tile> tiling(const cv::Mat& image, int size, int step) const;

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

    std::vector<WBFCluster> wbf(
        const std::vector<SegDetection>& detections,
        int image_w, int image_h,
        float iou_thr = 0.1f,
        float skip_thr = 0.01f) const;

    std::vector<SegDetection> delete_duplicates(const std::vector<SegDetection>& detections, const float iou_thresh_ = 0.05f) const;

    //statistische sort nach größe
    std::vector<std::vector<SegDetection>> sort_detections(const std::vector<SegDetection>& detections) const;
    
    //sort nach farben
    std::vector<int> sort_by_color(const cv::Mat& image, const std::vector<SegDetection>& detections, double percent = 15.0) const;

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

    //static constexpr int kPaletteSize = 1;
    //static const cv::Scalar kPalette[kPaletteSize];
};