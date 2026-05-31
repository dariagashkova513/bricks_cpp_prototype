#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <array>
#include <opencv2/opencv.hpp>

class LabHistogram {
public:
    static constexpr int L_BINS = 20;
    static constexpr int A_BINS = 16;
    static constexpr int B_BINS = 16;
    static constexpr int TOTAL = L_BINS * A_BINS * B_BINS;

    explicit LabHistogram(std::string path)
        : path_(std::move(path)), hist_(TOTAL, 0) {}

    void accumulate(const cv::Mat& labImage, const cv::Mat& mask) {
        CV_Assert(labImage.type() == CV_8UC3);
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(labImage.size() == mask.size());

        for (int y = 0; y < labImage.rows; y++) {
            for (int x = 0; x < labImage.cols; x++) {
                if (mask.at<uchar>(y, x) == 0) continue;
                cv::Vec3b px = labImage.at<cv::Vec3b>(y, x);

                int li = std::min(static_cast<int>(px[0] * L_BINS / 256), L_BINS - 1);
                int ai = std::min(static_cast<int>(px[1] * A_BINS / 256), A_BINS - 1);
                int bi = std::min(static_cast<int>(px[2] * B_BINS / 256), B_BINS - 1);

                hist_[li * A_BINS * B_BINS + ai * B_BINS + bi]++;
            }
        }

        write();
    }

    // Returns the dominant Lab color across all accumulated frames
    std::array<double, 3> dominantColor() const {
        int peakIdx = std::max_element(hist_.begin(), hist_.end()) - hist_.begin();

        int li = peakIdx / (A_BINS * B_BINS);
        int ai = (peakIdx / B_BINS) % A_BINS;
        int bi = peakIdx % B_BINS;

        return {
            ((li + 0.5) / L_BINS) * 100.0,
            ((ai + 0.5) / A_BINS) * 256.0 - 128.0,
            ((bi + 0.5) / B_BINS) * 256.0 - 128.0
        };
    }

    void reset() {
        std::fill(hist_.begin(), hist_.end(), 0);
        write();
    }

private:
    std::string      path_;
    std::vector<int> hist_;

    void write() const {
        std::ofstream f(path_);
        if (!f.is_open()) return;

        // Header
        f << "li ai bi count L_centre a_centre b_centre\n";

        for (int i = 0; i < TOTAL; i++) {
            if (hist_[i] == 0) continue;

            int li = i / (A_BINS * B_BINS);
            int ai = (i / B_BINS) % A_BINS;
            int bi = i % B_BINS;

            double L = ((li + 0.5) / L_BINS) * 100.0;
            double a = ((ai + 0.5) / A_BINS) * 256.0 - 128.0;
            double b = ((bi + 0.5) / B_BINS) * 256.0 - 128.0;

            f << li << " " << ai << " " << bi << " "
                << hist_[i] << " "
                << L << " " << a << " " << b << "\n";
        }
    }
};