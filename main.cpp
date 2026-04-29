#include "include/yolov8_seg_pt.hpp"
#include <iostream>

static const std::vector<std::string> COCO_CLASSES = {
    "brick"
};


//TODO: add performance metrics for testing
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

    std::cout << "Found " << detections.size() << " detection(s):\n";
    for (const auto& d : detections) {
        const std::string& name = d.class_id < (int)COCO_CLASSES.size()
            ? COCO_CLASSES[d.class_id] : "cls" + std::to_string(d.class_id);
        std::cout << "  [" << name << "]  conf=" << d.confidence
                  << "  box=(" << d.box.x << "," << d.box.y
                  << "," << d.box.width << "," << d.box.height << ")\n";
    }

    cv::Mat annotated = detector.draw(image, detections, COCO_CLASSES);
    cv::imwrite("output_seg1_nolabel.jpg", annotated);
    std::cout << "Saved annotated image to output_seg1_nolabel.jpg\n";
    return 0;
}


/*#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <filesystem>
#include <stdexcept>

#include <opencv2/opencv.hpp>

// ============================================================
//  Tile stub
// ============================================================

struct Tile {
    cv::Mat image;
    int     origin_x;
    int     origin_y;
    int     valid_w;
    int     valid_h;
};

// ============================================================
//  extract_tiles — standalone copy for testing
// ============================================================

static std::vector<Tile> extract_tiles(const cv::Mat& src,
    int            tile_size,
    int            step)
{
    std::vector<Tile> tiles;

    for (int y = 0; y < src.rows; y += step) {
        for (int x = 0; x < src.cols; x += step) {

            const int valid_w = std::min(tile_size, src.cols - x);
            const int valid_h = std::min(tile_size, src.rows - y);

            cv::Mat canvas(tile_size, tile_size, src.type(), cv::Scalar(114, 114, 114));
            src(cv::Rect(x, y, valid_w, valid_h))
                .copyTo(canvas(cv::Rect(0, 0, valid_w, valid_h)));

            tiles.push_back({ canvas, x, y, valid_w, valid_h });
        }
    }

    return tiles;
}

// ============================================================
//  Helpers
// ============================================================

static void print_tile_info(const Tile& t, int idx)
{
    std::cout << "  tile[" << idx << "]"
        << "  origin=(" << t.origin_x << "," << t.origin_y << ")"
        << "  valid=(" << t.valid_w << "x" << t.valid_h << ")"
        << "  canvas=" << t.image.cols << "x" << t.image.rows << "\n";
}

static int count_origins(int img_dim, int step)
{
    int n = 0;
    for (int v = 0; v < img_dim; v += step) ++n;
    return n;
}

// Ask the user yes/no on stdin — returns true for 'y'/'Y'
static bool ask_yes_no(const std::string& question)
{
    std::cout << question << " [y/n]: ";
    char c = 'n';
    std::cin >> c;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return (c == 'y' || c == 'Y');
}

// Ask for a string value with a default
static std::string ask_string(const std::string& question, const std::string& default_val)
{
    std::cout << question << " [default: " << default_val << "]: ";
    std::string line;
    std::getline(std::cin, line);
    return line.empty() ? default_val : line;
}

// ============================================================
//  Tests — all derived from the loaded image
// ============================================================

static bool test_tile_count(const cv::Mat& img, int tile_size, int step)
{
    std::cout << "[test_tile_count]\n";

    auto tiles = extract_tiles(img, tile_size, step);

    const int exp_cols = count_origins(img.cols, step);
    const int exp_rows = count_origins(img.rows, step);
    const int expected = exp_cols * exp_rows;

    std::cout << "  tile grid: " << exp_cols << " cols x " << exp_rows << " rows"
        << "  expected=" << expected << "  got=" << tiles.size() << "\n";

    assert(static_cast<int>(tiles.size()) == expected && "tile count mismatch");
    std::cout << "  PASS\n\n";
    return true;
}

static bool test_canvas_size(const cv::Mat& img, int tile_size, int step)
{
    std::cout << "[test_canvas_size]\n";

    auto tiles = extract_tiles(img, tile_size, step);

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].image.cols != tile_size || tiles[i].image.rows != tile_size) {
            std::cout << "  FAIL tile[" << i << "] canvas="
                << tiles[i].image.cols << "x" << tiles[i].image.rows
                << "  expected " << tile_size << "x" << tile_size << "\n";
            assert(false && "canvas size wrong");
        }
    }

    std::cout << "  All " << tiles.size() << " tiles are "
        << tile_size << "x" << tile_size << "  PASS\n\n";
    return true;
}

static bool test_padding_colour(int tile_size)
{
    std::cout << "[test_padding_colour]\n";

    // Use a synthetic image smaller than one tile — independent of input image
    const int small_w = std::max(1, tile_size / 2);
    const int small_h = std::max(1, tile_size / 2);
    std::cout << "  using synthetic " << small_w << "x" << small_h
        << " image (half of tile_size=" << tile_size << ")\n";

    cv::Mat synthetic(small_h, small_w, CV_8UC3, cv::Scalar(0, 255, 0));
    auto tiles = extract_tiles(synthetic, tile_size, tile_size);

    assert(tiles.size() == 1 && "expected exactly 1 tile");

    const cv::Mat& canvas = tiles[0].image;
    const int       last = tile_size - 1;

    const cv::Vec3b pad_px = canvas.at<cv::Vec3b>(last, last);
    const cv::Vec3b content_px = canvas.at<cv::Vec3b>(0, 0);

    std::cout << "  pad pixel     at (" << last << "," << last << "): "
        << (int)pad_px[0] << "," << (int)pad_px[1] << "," << (int)pad_px[2] << "\n";
    std::cout << "  content pixel at (0,0): "
        << (int)content_px[0] << "," << (int)content_px[1] << "," << (int)content_px[2] << "\n";

    assert(pad_px == cv::Vec3b(114, 114, 114) && "padding colour wrong");
    assert(content_px == cv::Vec3b(0, 255, 0) && "content colour wrong");

    std::cout << "  PASS\n\n";
    return true;
}

static bool test_origin_coords(const cv::Mat& img, int tile_size, int step)
{
    std::cout << "[test_origin_coords]\n";

    auto tiles = extract_tiles(img, tile_size, step);

    assert(!tiles.empty());
    assert(tiles.front().origin_x == 0 && tiles.front().origin_y == 0
        && "first tile must start at (0,0)");

    bool ok = true;
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].origin_x % step != 0 || tiles[i].origin_y % step != 0) {
            std::cout << "  BAD origin at tile[" << i << "]: ("
                << tiles[i].origin_x << "," << tiles[i].origin_y << ")\n";
            ok = false;
        }
    }

    // Print first row only
    const int first_row = count_origins(img.cols, step);
    std::cout << "  first row (" << first_row << " tiles):\n";
    for (int i = 0; i < std::min(first_row, static_cast<int>(tiles.size())); ++i)
        print_tile_info(tiles[i], i);

    assert(ok && "origin not a multiple of step");
    std::cout << "  PASS\n\n";
    return true;
}

static bool test_content_copied_correctly(const cv::Mat& img, int tile_size, int step)
{
    std::cout << "[test_content_copied_correctly]\n";

    // Build a synthetic image the same size as input but with encoded pixel coords
    cv::Mat encoded(img.rows, img.cols, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            encoded.at<cv::Vec3b>(y, x) = {
                static_cast<uchar>(x % 256),
                static_cast<uchar>(y % 256),
                static_cast<uchar>((x + y) % 256)
        };

    auto tiles = extract_tiles(encoded, tile_size, step);

    bool ok = true;
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        const int   lx = t.valid_w / 2;
        const int   ly = t.valid_h / 2;
        const int   gx = t.origin_x + lx;
        const int   gy = t.origin_y + ly;

        const cv::Vec3b expected = encoded.at<cv::Vec3b>(gy, gx);
        const cv::Vec3b got = t.image.at<cv::Vec3b>(ly, lx);

        if (expected != got) {
            std::cout << "  FAIL tile[" << i << "]"
                << " local=(" << lx << "," << ly << ")"
                << " global=(" << gx << "," << gy << ")"
                << " expected=(" << (int)expected[0] << ","
                << (int)expected[1] << ","
                << (int)expected[2] << ")"
                << " got=(" << (int)got[0] << ","
                << (int)got[1] << ","
                << (int)got[2] << ")\n";
            ok = false;
        }
    }

    assert(ok && "pixel content mismatch");
    std::cout << "  Checked " << tiles.size() << " tiles  PASS\n\n";
    return true;
}

static bool test_valid_dims_at_borders(const cv::Mat& img, int tile_size, int step)
{
    std::cout << "[test_valid_dims_at_borders]\n";

    auto tiles = extract_tiles(img, tile_size, step);

    bool ok = true;
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        const int   exp_vw = std::min(tile_size, img.cols - t.origin_x);
        const int   exp_vh = std::min(tile_size, img.rows - t.origin_y);
        const bool  border = (t.valid_w < tile_size || t.valid_h < tile_size);

        if (t.valid_w != exp_vw || t.valid_h != exp_vh) {
            std::cout << "  FAIL tile[" << i << "]"
                << " valid=(" << t.valid_w << "x" << t.valid_h << ")"
                << " expected=(" << exp_vw << "x" << exp_vh << ")\n";
            ok = false;
        }
        if (t.image.cols != tile_size || t.image.rows != tile_size) {
            std::cout << "  FAIL tile[" << i << "] canvas="
                << t.image.cols << "x" << t.image.rows
                << " expected " << tile_size << "x" << tile_size << "\n";
            ok = false;
        }
        if (border)
            print_tile_info(t, static_cast<int>(i));
    }

    assert(ok && "border tile dimensions wrong");
    std::cout << "  PASS\n\n";
    return true;
}

// ============================================================
//  Save tiles — asks the user interactively
// ============================================================

static void save_tiles(const cv::Mat& img, int tile_size, int step)
{
    const auto tiles = extract_tiles(img, tile_size, step);
    const int  n_cols = count_origins(img.cols, step);
    const int  n_total = static_cast<int>(tiles.size());

    std::cout << "\n--- Save tiles ---\n";
    std::cout << "  Total tiles: " << n_total << "\n";

    // Ask where to save
    const std::string out_dir = ask_string("  Output directory", "tile_debug");
    std::filesystem::create_directories(out_dir);

    // Ask how many to save
    bool save_all = ask_yes_no("  Save all " + std::to_string(n_total) + " tiles?");

    auto save_one = [&](int idx) {
        const std::string path = out_dir + "/tile_" + std::to_string(idx) + ".png";
        if (!cv::imwrite(path, tiles[idx].image))
            std::cerr << "  [warn] failed to write " << path << "\n";
        else
            std::cout << "  saved " << path
            << "  origin=(" << tiles[idx].origin_x << "," << tiles[idx].origin_y << ")"
            << "  valid=(" << tiles[idx].valid_w << "x" << tiles[idx].valid_h << ")\n";
    };

    if (save_all) {
        std::cout << "  Saving all " << n_total << " tiles...\n";
        for (int i = 0; i < n_total; ++i)
            save_one(i);
    }
    else {
        // Default: first row + last row
        std::cout << "  Saving first row (" << n_cols << " tiles) + last row...\n";
        for (int i = 0; i < std::min(n_cols, n_total); ++i)
            save_one(i);
        const int last_row_start = n_total - n_cols;
        for (int i = std::max(last_row_start, n_cols); i < n_total; ++i)
            save_one(i);
    }

    std::cout << "  Done. Saved to: "
        << std::filesystem::absolute(out_dir).string() << "\n";
}

// ============================================================
//  main
// ============================================================

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n"
            << "Example: " << argv[0] << " brick_wall.jpg\n";
        return 1;
    }

    // --- load image, detect size automatically ---
    const std::string img_path = argv[1];
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Error: could not load image: " << img_path << "\n";
        return 1;
    }

    const int img_w = img.cols;
    const int img_h = img.rows;

    std::cout << "=== extract_tiles tests ===\n\n";
    std::cout << "  image:     " << img_path << "\n";
    std::cout << "  size:      " << img_w << "x" << img_h << "\n";

    // --- tile parameters ---
    const int tile_size = 640;
    const int step = static_cast<int>(tile_size * 0.8f);  // 20% overlap

    std::cout << "  tile_size: " << tile_size << "\n";
    std::cout << "  step:      " << step << "\n";
    std::cout << "  tiles:     "
        << count_origins(img_w, step) << " cols x "
        << count_origins(img_h, step) << " rows = "
        << count_origins(img_w, step) * count_origins(img_h, step) << " total\n\n";

    // --- run tests ---
    try {
        test_tile_count(img, tile_size, step);
        test_canvas_size(img, tile_size, step);
        test_padding_colour(tile_size);           // synthetic — independent of input
        test_origin_coords(img, tile_size, step);
        test_content_copied_correctly(img, tile_size, step);
        test_valid_dims_at_borders(img, tile_size, step);
    }
    catch (const std::exception& e) {
        std::cerr << "\nFAIL: " << e.what() << "\n";
        return 1;
    }

    std::cout << "=== All tests passed ===\n";

    // --- ask whether to save tiles ---
    if (ask_yes_no("\nSave tiles to disk?"))
        save_tiles(img, tile_size, step);

    return 0;
}

*/