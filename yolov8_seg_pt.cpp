#include "include/yolov8_seg_pt.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <filesystem>

//=============================================================
// TODO
// 1 tiling Algorithmus
// 2 1 großes Bild mit Tile ersetzen
// 3 test auf geri nord jetzt
// 4 erneut trainierte model einsetzen
// 5 mehrere tests
//=============================================================



// ============================================================
//  Colour palette
// ============================================================

const cv::Scalar YOLOv8Seg::kPalette[kPaletteSize] = {
    {255,  128,  256}
};


static Ort::SessionOptions make_session_opts()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return opts;
}
// ============================================================
//  Constructor
// ============================================================

YOLOv8Seg::YOLOv8Seg(const std::string& model_path,
    float conf_thresh,
    float iou_thresh,
    int   input_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Seg"),
      session_opts_(make_session_opts()),
      session_(env_,
      std::filesystem::path(model_path).wstring().c_str(),
      session_opts_),
      input_size_(input_size),
      conf_thresh_(conf_thresh),
      iou_thresh_(iou_thresh)
{
    // Input name
    { auto r = session_.GetInputNameAllocated(0, allocator_);  input_name_   = r.get(); }
    // Two outputs: detection tensor + prototype masks
    { auto r = session_.GetOutputNameAllocated(0, allocator_); output0_name_ = r.get(); }
    { auto r = session_.GetOutputNameAllocated(1, allocator_); output1_name_ = r.get(); }
}


std::vector<Tile> YOLOv8Seg::tiling(const cv::Mat& image, int size=640, int step=512)const
{   
    std::vector<Tile> tiles;
    if (image.rows==size && image.cols==size){
        tiles.push_back({image.clone(), 0, 0, size, size});
        std::cout << "return the tile early, the size passes\n";
        return tiles;
    }

    for (int y = 0; y < image.rows; y += step) {
        for (int x = 0; x < image.cols; x += step) { 

            const int valid_w = std::min(size, image.cols - x);
            const int valid_h = std::min(size, image.rows - y);

            // Gray canvas — same pad colour as preprocess
            cv::Mat canvas(size, size, image.type(), cv::Scalar(114, 114, 114));
            image(cv::Rect(x, y, valid_w, valid_h))
                .copyTo(canvas(cv::Rect(0, 0, valid_w, valid_h)));

            std::cout << "cut out the tile #"<<y<<"\n";

            tiles.push_back({ canvas, x, y, valid_w, valid_h });
        }
    }

    return tiles;

}



// ============================================================
//  preprocess  — letterbox + NCHW float32
// ============================================================

cv::Mat YOLOv8Seg::preprocess(const cv::Mat& image,
                               float& scale, int& pad_w, int& pad_h) const
{
    scale = std::min(static_cast<float>(input_size_) / image.cols,
                     static_cast<float>(input_size_) / image.rows);

    const int new_w = static_cast<int>(std::round(image.cols * scale));
    const int new_h = static_cast<int>(std::round(image.rows * scale));
    pad_w = (input_size_ - new_w) / 2;
    pad_h = (input_size_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_w, pad_h, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    return cv::dnn::blobFromImage(rgb);
}

// ============================================================
//  detect
// ============================================================

std::vector<SegDetection> YOLOv8Seg::detect(
    const cv::Mat& image
)
{
    if (image.empty())
        throw std::runtime_error("YOLOv8Seg::detect — empty image");

 

    const int orig_w = image.cols;
    const int orig_h = image.rows;

    const auto tiles = tiling(image);
    std::vector<SegDetection> all_detections;

    for (const auto& tile : tiles) {
        std::cout << "[Inference] Processing tile at (" << tile.origin_x
            << ", " << tile.origin_y << ") size: " << tile.valid_w
            << "x" << tile.valid_h << "..." << std::endl;


        float scale = 0.f;
        int pad_w, pad_h = 0;
        cv::Mat blob = preprocess(tile.image, scale, pad_w, pad_h);

        // Build input tensor
        const std::array<int64_t, 4> input_shape{ 1, 3, input_size_, input_size_ };
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            reinterpret_cast<float*>(blob.data),
            static_cast<size_t>(blob.total()),
            input_shape.data(), input_shape.size());

        const char* input_names[] = { input_name_.c_str() };
        const char* output_names[] = { output0_name_.c_str(), output1_name_.c_str() };

        auto outputs = session_.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1,
            output_names, 2);

        // ---- output0: [1, 4+nc+nm, na] ----
        auto shape0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        // shape0 = [1, rows, num_anchors]
        // rows   = 4 (box) + num_classes + nm (mask coeffs, usually 32)
        const int64_t num_anchors = shape0[2];

        // ---- output1: [1, nm, proto_h, proto_w] ----
        auto shape1 = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const int64_t nm = shape1[1];
        const int     proto_h = static_cast<int>(shape1[2]);
        const int     proto_w = static_cast<int>(shape1[3]);

        const int64_t num_classes = shape0[1] - 4 - nm;

        const float* det_data = outputs[0].GetTensorData<float>();
        const float* proto_data = outputs[1].GetTensorData<float>();

        // --- postprocess in tile-local coords ---
        auto dets = postprocess(det_data, num_anchors, num_classes,
            proto_data, nm, proto_h, proto_w,
            scale, pad_w, pad_h,
            tile.valid_w, tile.valid_h);

        //maybe get the couple of tiles' visualisaztion to see

        // --- remap boxes to global coords, store mask origin ---
        for (auto& det : dets) {
            det.box.x += tile.origin_x;
            det.box.y += tile.origin_y;
            det.mask_origin = { tile.origin_x, tile.origin_y };
        }

        all_detections.insert(all_detections.end(),
            std::make_move_iterator(dets.begin()),
            std::make_move_iterator(dets.end()));
    }

    // --- second NMS pass to remove cross-tile duplicates ---
    std::cout << "started second NMS round ... \n";
    std::vector<cv::Rect> all_boxes;
    std::vector<float>    all_scores;
    all_boxes.reserve(all_detections.size());
    all_scores.reserve(all_detections.size());

    for (const auto& d : all_detections) {
        all_boxes.push_back(d.box);
        all_scores.push_back(d.confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(all_boxes, all_scores, conf_thresh_, iou_thresh_, keep);

    std::vector<SegDetection> final_detections;
    final_detections.reserve(keep.size());
    for (int i : keep)
        final_detections.push_back(std::move(all_detections[i]));
    
    std::cout << "finished second NMS round \n";
    return final_detections;

}

// ============================================================
//  assembleMask
//
//  mask = sigmoid( proto [nm x proto_h x proto_w]  ·  coeffs [nm] )
//  then crop to the detection box and resize to original image dims.
// ============================================================

cv::Mat YOLOv8Seg::assembleMask(const float*    proto_data,
                                 int64_t         nm,
                                 int             proto_h,
                                 int             proto_w,
                                 const float*    coeffs,
                                 const cv::Rect& box,
                                 int             orig_w,
                                 int             orig_h,
                                 float           scale,
                                 int             pad_w,
                                 int             pad_h) const
{
    // 1. Linear combination of prototype masks
    //    proto_data layout: [nm, proto_h, proto_w]  (flat, row-major)
    cv::Mat combined(proto_h, proto_w, CV_32F, 0.f);
    for (int64_t k = 0; k < nm; ++k) {
        // k-th prototype: starts at proto_data + k * proto_h * proto_w
        cv::Mat proto_k(proto_h, proto_w, CV_32F,
                        const_cast<float*>(proto_data) + k * proto_h * proto_w);
        combined += coeffs[k] * proto_k;
    }

    // 2. Sigmoid activation
    cv::exp(-combined, combined);            // combined = e^(-x)
    combined = 1.f / (1.f + combined);       // sigmoid

    // 3. Map box from original-image space → proto space
    const float proto_scale_x = static_cast<float>(proto_w) / input_size_;
    const float proto_scale_y = static_cast<float>(proto_h) / input_size_;

    const float lx1 = box.x * scale + pad_w;
    const float ly1 = box.y * scale + pad_h;
    const float lx2 = (box.x + box.width) * scale + pad_w;
    const float ly2 = (box.y + box.height) * scale + pad_h;

    const int px1 = std::clamp(static_cast<int>(lx1 * proto_scale_x), 0, proto_w - 1);
    const int py1 = std::clamp(static_cast<int>(ly1 * proto_scale_y), 0, proto_h - 1);
    const int px2 = std::clamp(static_cast<int>(std::ceil(lx2 * proto_scale_x)), px1 + 1, proto_w);
    const int py2 = std::clamp(static_cast<int>(std::ceil(ly2 * proto_scale_y)), py1 + 1, proto_h);

    // 4. Crop proto region and resize to box dimensions
    cv::Mat cropped = combined(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();

    // Clamp box to tile bounds before resizing
    const int safe_x = std::clamp(box.x, 0, orig_w - 1);
    const int safe_y = std::clamp(box.y, 0, orig_h - 1);
    const int safe_w = std::min(box.width, orig_w - safe_x);
    const int safe_h = std::min(box.height, orig_h - safe_y);

    if (safe_w <= 0 || safe_h <= 0)
        return cv::Mat();   // degenerate box — return empty mask

    cv::Mat resized_mask;
    cv::resize(cropped, resized_mask, { safe_w, safe_h }, 0, 0, cv::INTER_LINEAR);

    // 5. Threshold → binary CV_8U, sized to box
    cv::Mat binary;
    cv::threshold(resized_mask, binary, 0.5, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    return binary;  // tile-local, placed by draw() using mask_origin
}

// ============================================================
//  postprocess
// ============================================================

std::vector<SegDetection>
YOLOv8Seg::postprocess(const float* det_data,   int64_t num_anchors, int64_t num_classes,
                        const float* proto_data, int64_t nm,
                        int proto_h,             int proto_w,
                        float scale,             int pad_w, int pad_h,
                        int orig_w,              int orig_h) const
{
    // Layout of det_data: [4+nc+nm, num_anchors] (row-major)
    //   rows 0-3      : cx, cy, w, h
    //   rows 4..4+nc-1: class scores
    //   rows 4+nc..   : nm mask coefficients

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       class_ids;
    std::vector<std::vector<float>> mask_coeffs_list;

    //const int64_t total_rows = 4 + num_classes + nm;

    for (int64_t a = 0; a < num_anchors; ++a) {
        // Best class score
        int   best_cls   = 0;
        float best_score = 0.f;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float s = det_data[(4 + c) * num_anchors + a];
            if (s > best_score) { best_score = s; best_cls = static_cast<int>(c); }
        }
        if (best_score < conf_thresh_) continue;

        // Box (centre format, letterbox space)
        const float cx = det_data[0 * num_anchors + a];
        const float cy = det_data[1 * num_anchors + a];
        const float bw = det_data[2 * num_anchors + a];
        const float bh = det_data[3 * num_anchors + a];

        // Unpad and unscale
        auto unpad = [&](float v, float pad, float sc, int limit) -> int {
            return std::clamp(static_cast<int>((v - pad) / sc), 0, limit - 1);
        };

        const int x1 = unpad(cx - bw / 2.f, static_cast<float>(pad_w), scale, orig_w);
        const int y1 = unpad(cy - bh / 2.f, static_cast<float>(pad_h), scale, orig_h);
        const int x2 = unpad(cx + bw / 2.f, static_cast<float>(pad_w), scale, orig_w);
        const int y2 = unpad(cy + bh / 2.f, static_cast<float>(pad_h), scale, orig_h);

        // Mask coefficients
        std::vector<float> coeffs(nm);
        for (int64_t k = 0; k < nm; ++k)
            coeffs[k] = det_data[(4 + num_classes + k) * num_anchors + a];

        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(best_score);
        class_ids.push_back(best_cls);
        mask_coeffs_list.push_back(std::move(coeffs));
    }

    // NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, iou_thresh_, nms_indices);

    std::vector<SegDetection> detections;
    detections.reserve(nms_indices.size());

    for (int idx : nms_indices) {
        SegDetection d;
        d.class_id   = class_ids[idx];
        d.confidence = scores[idx];
        d.box        = boxes[idx];
        d.mask       = assembleMask(proto_data, nm, proto_h, proto_w,
                                    mask_coeffs_list[idx].data(),
                                    d.box, orig_w, orig_h,
                                    scale, pad_w, pad_h);
        detections.push_back(std::move(d));
    }

    return detections;
}

// ============================================================
//  draw
// ============================================================

cv::Mat YOLOv8Seg::draw(const cv::Mat&                   image,
                         const std::vector<SegDetection>& detections,
                         const std::vector<std::string>&  class_names,
                         float                            alpha) const
{
    cv::Mat out = image.clone();


    // Allocate full-image buffers once and reuse per detection
    cv::Mat full_mask = cv::Mat::zeros(image.size(), CV_8U);
    cv::Mat colour_mask = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat mask_overlay = cv::Mat::zeros(image.size(), CV_8UC3);

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d      = detections[i];
        const cv::Scalar& colour = kPalette[i % kPaletteSize];

        // --- mask overlay ---
        if (!d.mask.empty()) {
            // 1. Create a region of interest (ROI) in the output image based on the box
            // We must intersect the box with the image boundaries to avoid crashes
            cv::Rect image_rect(0, 0, out.cols, out.rows);
            cv::Rect roi_rect = d.box & image_rect;

            if (roi_rect.width > 0 && roi_rect.height > 0) {
                // 2. The mask might be slightly larger/smaller than the ROI if clipped
                // Ensure the mask is resized to the exact ROI size
                cv::Mat actual_mask;
                if (d.mask.size() != roi_rect.size()) {
                    cv::resize(d.mask, actual_mask, roi_rect.size());
                }
                else {
                    actual_mask = d.mask;
                }

                // 3. Create the colored overlay for this specific ROI
                cv::Mat roi_img = out(roi_rect);
                cv::Mat color_roi(roi_rect.size(), CV_8UC3, colour);

                // 4. Blend only where the mask is positive
                // Using cv::addWeighted on the ROI directly is more efficient
                cv::Mat masked_overlay;
                color_roi.copyTo(masked_overlay, actual_mask);
                cv::addWeighted(roi_img, 1.0, masked_overlay, alpha, 0.0, roi_img);
            }
        }

        // --- bounding box ---
        cv::rectangle(out, d.box, colour, 2);
        /*
        // --- label ---
        std::string label = class_names.empty()
            ? "cls" + std::to_string(d.class_id)
            : class_names[d.class_id];
        label += " " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";

        int baseline = 0;
        auto tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::Point tl{d.box.x, std::max(d.box.y - tsz.height - 4, 0)};
        cv::rectangle(out, tl,
                       {tl.x + tsz.width, tl.y + tsz.height + baseline + 4},
                       colour, cv::FILLED);
        cv::putText(out, label, {d.box.x, tl.y + tsz.height},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);*/
    }

    return out;
}