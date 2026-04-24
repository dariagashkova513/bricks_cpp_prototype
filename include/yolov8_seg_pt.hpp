#ifndef YOLOV8_PT_HPP
#define YOLOV8_PT_HPP

#include <fstream>
#include <vector>
#include <string>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime>  //find the right headers


struct Segmenation{
    std::vector<float> mask;
    float confidence;
    int class_id;
}

class yolov8_seg_pt
{
private:
    cv:: Mat preprocess(const cv::Mat& image);
    std::vector<Segmentation> postprocess(const float*  output,
                                       int64_t        num_dets,
                                       int64_t        num_classes,
                                       float          scale,
                                       int            pad_w,
                                       int            pad_h,
                                       int            orig_w,
                                       int            orig_h) const;
    
    //onnxruntime variables
    Ort::Env env;
    Ort::Session* session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    //model things

    
public:
    yolov8_seg_pt(/* args */);
    ~yolov8_seg_pt();
    explicit yolov8_seg_pt(const std::string& model_path
                            float conf_threshold = 0.25,
                            float iou_threshold = 0.45,
                            int img_size = 1024);

    std::vector<Segmenation> infer(const cv::Mat& image);

    cv::Mat visualize(const cv::Mat& image,
                        const std::vector<Segmenation>& segmentations,
                        const std::vector<std::string>& class_names  = {}) const;
};

yolov8_seg_pt::yolov8_seg_pt(/* args */)
{
}

yolov8_seg_pt::~yolov8_seg_pt()
{
}

#endif