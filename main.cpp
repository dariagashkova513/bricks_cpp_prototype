#include "include/yolov8_seg_pt.hpp"
#include "include/color_space_cell.hpp"
#include <iostream>
#include <array>

static const std::vector<std::string> COCO_CLASSES = {
    "brick"
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model-seg.onnx> <image.(jpg|png)> [conf=0.75] [iou=0.15]\n";
        return 1;
    }

    const float conf = argc > 3 ? std::stof(argv[3]) : 0.25f;
    const float iou  = argc > 4 ? std::stof(argv[4]) : 0.45f;

    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) { std::cerr << "Cannot open: " << argv[2] << "\n"; return 1; }

    YOLOv8Seg detector(argv[1], conf, iou, 640);
    auto detections = detector.detect(image);

    //std::cout << "Found " << detections.size() << " detection(s):\n";
    for (const auto& d : detections) {
        const std::string& name = d.class_id < (int)COCO_CLASSES.size()
            ? COCO_CLASSES[d.class_id] : "cls" + std::to_string(d.class_id);
        //std::cout << "  [" << name << "]  conf=" << d.confidence
        //          << "  box=(" << d.box.x << "," << d.box.y
        //          << "," << d.box.width << "," << d.box.height << ")\n";
    }

    cv::Mat annotated = detector.draw(image, detections, COCO_CLASSES);
    cv::imwrite("output_seg1_nolabel.jpg", annotated);
    std::cout << "Saved annotated image to output_seg1_nolabel.jpg\n";
    return 0;
}

/*
int main(int argc, char* argv[]) {

    const double pr = std::stof(argv[1]);

    DynamicGrid dgrid(pr);

    std::array<double, 3> point = { 13.9998, 81.2123, 150.56 };
    dgrid.getDCSS(point);

    return 0;
}*/