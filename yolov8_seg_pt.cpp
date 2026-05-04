#include "include/yolov8_seg_pt.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <filesystem>

//=============================================================
// TODO
// 1 tiling Algorithmus : BEARBEITEN
    /**  1. Use step = 320 (50 % overlap)
    2. Predict on all tiles normally
    3. Convert all masks to global coordinates
    4. Use WBF to merge boxes → get cluster membership per instance
    5. For each WBF cluster : union or weighted - blend all contributing masks
    6. Final threshold the blended mask*/
// 2 1 großes Bild mit Tile ersetzen /
// 3 test auf geri nord jetzt / 
// 4 erneut trainierte model einsetzen /
// 5 mehrere tests
//=============================================================

constexpr float CONF_PREPROCESS = 0.25f;
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

// ============================================================
//  Helper function for removing double masks
// 
// ============================================================
std::vector<SegDetection> YOLOv8Seg::delete_duplicates(const std::vector<SegDetection>& detections, const float iou_thresh_) const {
    if (detections.empty()) return {};

    const int n = static_cast<int>(detections.size());

    // ── IoU between two cv::Rect ─────────────────────────────
    auto rect_iou = [](const cv::Rect& a, const cv::Rect& b) -> float {
        const cv::Rect inter = a & b;
        if (inter.empty()) return 0.f;
        const float inter_area = static_cast<float>(inter.area());
        return inter_area / (a.area() + b.area() - inter_area);
    };

    // ── One box's center lies inside the other ───────────────
    auto center_in_box = [](const cv::Rect& a, const cv::Rect& b) -> bool {
        const cv::Point ca{ a.x + a.width / 2, a.y + a.height / 2 };
        const cv::Point cb{ b.x + b.width / 2, b.y + b.height / 2 };
        return b.contains(ca) || a.contains(cb);
    };

    // ── One box is almost entirely inside the other ──────────
    //   catches the "small partial mask inside full mask" case
    auto is_contained = [](const cv::Rect& small, const cv::Rect& large) -> bool {
        const cv::Rect inter = small & large;
        if (inter.empty()) return false;
        const float overlap = static_cast<float>(inter.area())
            / static_cast<float>(small.area());
        return overlap > 0.80f;   // 80% of smaller box covered by larger
    };

    // ── Sort indices by confidence descending ────────────────
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
        });

    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        if (suppressed[order[i]]) continue;

        const auto& di = detections[order[i]];

        for (int j = i + 1; j < n; ++j) {
            if (suppressed[order[j]]) continue;

            const auto& dj = detections[order[j]];
            if (di.class_id != dj.class_id) continue;

            const bool iou_dup = rect_iou(di.box, dj.box) > iou_thresh_;
            const bool center_dup = center_in_box(di.box, dj.box);
            const bool i_inside_j = is_contained(di.box, dj.box);  // di is the smaller
            const bool j_inside_i = is_contained(dj.box, di.box);  // dj is the smaller

            if (iou_dup || center_dup) {
                // Standard case — suppress lower confidence (j, already sorted)
                suppressed[order[j]] = true;
            }
            else if (i_inside_j) {
                // di is contained inside dj → di is the partial, suppress it
                // regardless of confidence
                suppressed[order[i]] = true;
                break;  // order[i] is gone, stop inner loop
            }
            else if (j_inside_i) {
                // dj is contained inside di → dj is the partial, suppress it
                suppressed[order[j]] = true;
            }
        }
    }

    // ── Collect survivors ────────────────────────────────────
    std::vector<SegDetection> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i)
        if (!suppressed[i])
            result.push_back(detections[i]);

    return result;
}


std::vector<Tile> YOLOv8Seg::tiling(const cv::Mat& image, int size=640, int step=320)const
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
    const cv::Mat& image,
    const float conf_postprocess
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

        dets.erase(std::remove_if(dets.begin(), dets.end(),
            [](const SegDetection& d) { return d.confidence < CONF_PREPROCESS; }),
            dets.end());

        // --- remove boxes closest to tile borders if border != 0


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

    /*std::vector<int> keep;
    cv::dnn::NMSBoxes(all_boxes, all_scores, conf_thresh_, iou_thresh_, keep);

    std::vector<SegDetection> final_detections;
    final_detections.reserve(keep.size());
    std::cout << "detections before nms" << keep.size() << "\n";
    for (int i : keep) {
        if (all_detections[i].confidence >= CONF_POSTPROCESS)   // ← only this loop
            final_detections.push_back(std::move(all_detections[i]));
    }
    
    std::cout << "finished second NMS round \n"
        << final_detections.size() << " detections\n";*/
        // ── Replace cv::dnn::NMSBoxes with WBF ───────────────────────
    auto clusters = wbf(all_detections, orig_w, orig_h,
        iou_thresh_,   // cluster IoU threshold
        0.01f);        // skip threshold

    std::vector<SegDetection> final_detections;
    final_detections.reserve(clusters.size());

    for (const auto& cl : clusters) {
        if (cl.fused_score < CONF_POSTPROCESS) continue;

        // Pick highest-confidence member as the base detection
        int best_idx = cl.members[0];
        float best_conf = 0.f;
        for (int idx : cl.members) {
            if (all_detections[idx].confidence > best_conf) {
                best_conf = all_detections[idx].confidence;
                best_idx = idx;
            }
        }

        SegDetection d = all_detections[best_idx];

        // ★ Use the WBF-fused box instead of the winner's box
        d.box = cv::Rect(
            static_cast<int>(cl.fused_box.x),
            static_cast<int>(cl.fused_box.y),
            static_cast<int>(cl.fused_box.width),
            static_cast<int>(cl.fused_box.height));
        d.confidence = cl.fused_score;

        // ★ Composite masks from all cluster members
        //   (union — any pixel positive in any member mask counts)
        if (cl.members.size() > 1) {
            cv::Mat combined = cv::Mat::zeros(d.box.size(), CV_8U);
            for (int idx : cl.members) {
                const auto& md = all_detections[idx];
                if (md.mask.empty()) continue;
                // Place member mask into fused box space
                cv::Rect member_roi = md.box & d.box;
                if (member_roi.width <= 0 || member_roi.height <= 0) continue;
                cv::Rect local_roi(
                    member_roi.x - d.box.x,
                    member_roi.y - d.box.y,
                    member_roi.width,
                    member_roi.height);
                cv::Rect src_roi(
                    member_roi.x - md.box.x,
                    member_roi.y - md.box.y,
                    member_roi.width,
                    member_roi.height);
                if (src_roi.width > md.mask.cols) src_roi.width = md.mask.cols;
                if (src_roi.height > md.mask.rows) src_roi.height = md.mask.rows;
                cv::Mat src_patch = md.mask(src_roi);
                cv::Mat dst_patch = combined(local_roi);
                cv::bitwise_or(dst_patch, src_patch, dst_patch);
            }
            d.mask = combined;
        }

        final_detections.push_back(std::move(d));
    }
    final_detections = delete_duplicates(final_detections);
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

cv::Mat YOLOv8Seg::draw(const cv::Mat& image,
    const std::vector<SegDetection>& detections,
    const std::vector<std::string>& class_names,
    float                            alpha) const
{
    cv::Mat out = image.clone();

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        const cv::Scalar& colour = kPalette[i % kPaletteSize];
        /*
        // --- mask overlay ---
        if (!d.mask.empty()) {
            cv::Rect image_rect(0, 0, out.cols, out.rows);
            cv::Rect roi_rect = d.box & image_rect;

            if (roi_rect.width > 0 && roi_rect.height > 0) {

                // 1. Resize mask to ROI if needed
                cv::Mat actual_mask;
                if (d.mask.size() != roi_rect.size())
                    cv::resize(d.mask, actual_mask, roi_rect.size(),
                        0, 0, cv::INTER_NEAREST);
                else
                    actual_mask = d.mask;

                // binarize the mask properly (d.mask is already
                //   uint8 after postprocess, skip this if already binary)
                cv::Mat binary_mask;
                if (actual_mask.type() == CV_32F)
                    binary_mask = actual_mask > 0.75f;   // float soft mask
                else
                    binary_mask = actual_mask > 127;    // uint8 mask

                // fill ratio guard, correct variable names
                double mask_area = cv::countNonZero(binary_mask);
                double box_area = static_cast<double>(roi_rect.width)
                    * static_cast<double>(roi_rect.height);
                if (box_area > 0 && (mask_area / box_area) < 0.05)
                    continue;   // mask is nearly empty — skip this detection

                // 2. Blend coloured overlay where mask is active
                cv::Mat roi_img = out(roi_rect);
                cv::Mat color_roi(roi_rect.size(), CV_8UC3, colour);

                cv::Mat masked_overlay = cv::Mat::zeros(roi_rect.size(), CV_8UC3);
                color_roi.copyTo(masked_overlay, binary_mask);
                cv::addWeighted(roi_img, 1.0, masked_overlay, alpha, 0.0, roi_img);
            }
        }
        */
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

// ============================================================
//  WBF  — Weighted Boxes Fusion (cross-tile global merge)
//  Fuses boxes from overlapping tiles instead of suppressing.
//  Returns clusters; caller composites masks from d.members.
// ============================================================
std::vector<WBFCluster> YOLOv8Seg::wbf(
    const std::vector<SegDetection>& dets,
    int image_w, int image_h,
    float iou_thr, float skip_thr) const
{
    // Normalise boxes to [0,1]
    const float iw = static_cast<float>(image_w);
    const float ih = static_cast<float>(image_h);

    struct NBox {
        float x1, y1, x2, y2, score;
        int   class_id, orig_idx;
    };

    // Filter and normalise
    std::vector<NBox> nboxes;
    nboxes.reserve(dets.size());
    for (int i = 0; i < (int)dets.size(); ++i) {
        if (dets[i].confidence < skip_thr) continue;
        const auto& b = dets[i].box;
        nboxes.push_back({
            b.x / iw,
            b.y / ih,
            (b.x + b.width) / iw,
            (b.y + b.height) / ih,
            dets[i].confidence,
            dets[i].class_id,
            i
            });
    }

    // Sort descending by score
    std::sort(nboxes.begin(), nboxes.end(),
        [](const NBox& a, const NBox& b) { return a.score > b.score; });

    // ── IoU helper ───────────────────────────────────────────
    auto iou = [](const NBox& a, const NBox& b) -> float {
        const float ix1 = std::max(a.x1, b.x1);
        const float iy1 = std::max(a.y1, b.y1);
        const float ix2 = std::min(a.x2, b.x2);
        const float iy2 = std::min(a.y2, b.y2);
        const float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
        if (inter == 0.f) return 0.f;
        const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (area_a + area_b - inter);
    };

    // ── Cluster boxes ────────────────────────────────────────
    // Each cluster accumulates weighted box coordinates
    struct Cluster {
        std::vector<NBox> boxes;   // all members
        // running weighted sums
        float wx1 = 0, wy1 = 0, wx2 = 0, wy2 = 0, wsum = 0;
    };
    std::vector<Cluster> clusters;

    for (const auto& nb : nboxes) {
        bool matched = false;
        for (auto& cl : clusters) {
            // Compare against the current fused box of the cluster
            NBox fused{
                cl.wx1 / cl.wsum, cl.wy1 / cl.wsum,
                cl.wx2 / cl.wsum, cl.wy2 / cl.wsum,
                0.f, nb.class_id, -1
            };
            if (nb.class_id == cl.boxes[0].class_id && iou(nb, fused) > iou_thr) {
                cl.boxes.push_back(nb);
                cl.wx1 += nb.score * nb.x1;
                cl.wy1 += nb.score * nb.y1;
                cl.wx2 += nb.score * nb.x2;
                cl.wy2 += nb.score * nb.y2;
                cl.wsum += nb.score;
                matched = true;
                break;
            }
        }
        if (!matched) {
            Cluster cl;
            cl.boxes.push_back(nb);
            cl.wx1 = nb.score * nb.x1;
            cl.wy1 = nb.score * nb.y1;
            cl.wx2 = nb.score * nb.x2;
            cl.wy2 = nb.score * nb.y2;
            cl.wsum = nb.score;
            clusters.push_back(std::move(cl));
        }
    }

    // ── Build output ─────────────────────────────────────────
    std::vector<WBFCluster> result;
    result.reserve(clusters.size());

    for (const auto& cl : clusters) {
        // Fused score = avg of top members (WBF paper formula)
        float fused_score = cl.wsum / static_cast<float>(cl.boxes.size());
        if (fused_score < skip_thr) continue;

        // Denormalise fused box back to pixel coords
        const float fx1 = (cl.wx1 / cl.wsum) * iw;
        const float fy1 = (cl.wy1 / cl.wsum) * ih;
        const float fx2 = (cl.wx2 / cl.wsum) * iw;
        const float fy2 = (cl.wy2 / cl.wsum) * ih;

        WBFCluster out;
        out.fused_box = { fx1, fy1, fx2 - fx1, fy2 - fy1 };
        out.fused_score = fused_score;
        out.fused_class = cl.boxes[0].class_id;
        for (const auto& nb : cl.boxes)
            out.members.push_back(nb.orig_idx);

        result.push_back(std::move(out));
    }

    return result;
}