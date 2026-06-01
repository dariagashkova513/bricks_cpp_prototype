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
        : path_(std::move(path)) {}

    void accumulate(const cv::Mat& labImage, const cv::Mat& mask, int detectionIdx) {
        CV_Assert(labImage.type() == CV_8UC3);
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(labImage.size() == mask.size());

        // Extend storage if needed
        if (detectionIdx >= static_cast<int>(hists_.size())) {
            hists_.resize(detectionIdx + 1);
            hists_[detectionIdx] = std::vector<int>(TOTAL, 0);
        }

        auto& hists = hists_[detectionIdx];
        for (int y = 0; y < labImage.rows; y++) {
            for (int x = 0; x < labImage.cols; x++) {
                if (mask.at<uchar>(y, x) == 0) continue;
                cv::Vec3b px = labImage.at<cv::Vec3b>(y, x);
                int li = std::min(static_cast<int>(px[0] * L_BINS / 256), L_BINS - 1);
                int ai = std::min(static_cast<int>(px[1] * A_BINS / 256), A_BINS - 1);
                int bi = std::min(static_cast<int>(px[2] * B_BINS / 256), B_BINS - 1);
                hists[li * A_BINS * B_BINS + ai * B_BINS + bi]++;
            }
        }
        write();
    }

    void reset() {
        hists_.clear();
        write();
    }

    // Returns the dominant Lab color across all accumulated frames
    std::array<double, 3> dominantColor(int detectionIdx) const {
        if (detectionIdx >= static_cast<int>(hists_.size())) return { 0.0, 0.0, 0.0 };
        const auto& hist = hists_[detectionIdx];
        int peakIdx = std::max_element(hist.begin(), hist.end()) - hist.begin();
        int li = peakIdx / (A_BINS * B_BINS);
        int ai = (peakIdx / B_BINS) % A_BINS;
        int bi = peakIdx % B_BINS;
        return {
            ((li + 0.5) / L_BINS) * 100.0,
            ((ai + 0.5) / A_BINS) * 256.0 - 128.0,
            ((bi + 0.5) / B_BINS) * 256.0 - 128.0
        };
    }

private:
    std::string      path_;
    std::vector<std::vector<int>> hists_;

    void write() const {
        std::ofstream f(path_);
        if (!f.is_open()) return;
        f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        f << "<histograms>\n";
        for (int d = 0; d < static_cast<int>(hists_.size()); d++) {
            f << "  <detection>\n";
            f << "    <detection_index>" << d << "</detection_index>\n";
            const auto& hist = hists_[d];
            for (int i = 0; i < TOTAL; i++) {
                if (hist[i] == 0) continue;
                int li = i / (A_BINS * B_BINS);
                int ai = (i / B_BINS) % A_BINS;
                int bi = i % B_BINS;
                f << "    <bin>\n";
                f << "      <li>" << li << "</li>\n";
                f << "      <ai>" << ai << "</ai>\n";
                f << "      <bi>" << bi << "</bi>\n";
                f << "      <count>" << hist[i] << "</count>\n";
                f << "    </bin>\n";
            }
            f << "  </detection>\n";
        }
        f << "</histograms>\n";
    }
};