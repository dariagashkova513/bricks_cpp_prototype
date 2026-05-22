#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

using Point3D = std::array<double, 3>;
using Vector3D = std::array<double, 3>;

struct DetectedColorSubSpace {
    Vector3D bounds;
    int id;
    std::string assigned_color;
};

class DynamicGrid {

public:

    static constexpr double HUE_AXIS = 179.0;
    static constexpr double SAT_VAL_AXIS = 255.0;

    explicit DynamicGrid(double percent) {
        if (percent <= 0.0 || percent > 100.0)
            throw std::invalid_argument("percent must be in the range (0, 100]");

        divisions_ = static_cast<int>(std::round(100.0 / percent));

        cell_size_[0] = HUE_AXIS / divisions_;
        cell_size_[1] = SAT_VAL_AXIS / divisions_;
        cell_size_[2] = SAT_VAL_AXIS / divisions_;

        origin_ = { 0.0, 0.0, 0.0 };
    };

    void isInBoundsOfnSubspace(const Point3D& point, int subspaceID) {
        const int total = divisions_ * divisions_ * divisions_;
        if (subspaceID < 0 || subspaceID >= total)
            throw std::out_of_range("subspaceID " + std::to_string(subspaceID) +
                " is out of range [0, " +
                std::to_string(total - 1) + "]");

        const int hi = subspaceID / (divisions_ * divisions_);
        const int si = (subspaceID / divisions_) % divisions_;
        const int vi = subspaceID % divisions_;

        const double h_min = origin_[0] + hi * cell_size_[0];
        const double s_min = origin_[1] + si * cell_size_[1];
        const double v_min = origin_[2] + vi * cell_size_[2];

        const bool inside = (point[0] >= h_min && point[0] < h_min + cell_size_[0]) &&
            (point[1] >= s_min && point[1] < s_min + cell_size_[1]) &&
            (point[2] >= v_min && point[2] < v_min + cell_size_[2]);

        if (!inside)
            throw std::domain_error(
                "Point (" + std::to_string(point[0]) + ", " +
                std::to_string(point[1]) + ", " +
                std::to_string(point[2]) +
                ") is not within subspace " + std::to_string(subspaceID));
    };
    
    [[nodiscard]]
    DetectedColorSubSpace getDCSS(std::array<double, 3> point) {
        point[0] = std::clamp(point[0], 0.0, HUE_AXIS);
        point[1] = std::clamp(point[1], 0.0, SAT_VAL_AXIS);
        point[2] = std::clamp(point[2], 0.0, SAT_VAL_AXIS);

        const int hi = std::min(static_cast<int>(point[0] / cell_size_[0]), divisions_ - 1);
        const int si = std::min(static_cast<int>(point[1] / cell_size_[1]), divisions_ - 1);
        const int vi = std::min(static_cast<int>(point[2] / cell_size_[2]), divisions_ - 1);

        const int id = hi * divisions_ * divisions_ + si * divisions_ + vi;

        DetectedColorSubSpace dcss;
        dcss.bounds = cell_size_;
        dcss.id = id;
        dcss.assigned_color = hsvCellToColorName(hi, si, vi);

        std::cout << dcss.id << "\n" << dcss.assigned_color << "\n";
        return dcss;
    }
    
private:
    int    divisions_;
    Vector3D cell_size_;
    Point3D  origin_;

    [[nodiscard]]
    std::string hsvCellToColorName(int hi, int si, int vi) const {
        // Normalise indices to [0, 1] using cell centres
        const double h = ((hi + 0.5) * cell_size_[0]) / HUE_AXIS;      // 0-1
        const double s = ((si + 0.5) * cell_size_[1]) / SAT_VAL_AXIS;  // 0-1
        const double v = ((vi + 0.5) * cell_size_[2]) / SAT_VAL_AXIS;  // 0-1

        std::string value_;
        
        // Low value : dark tones
        if (v < 0.20)
        {
            return "Black";
        }
        else if (v >= 0.20 && v < 0.40) {
            value_ = "Dark";
        }
        else if (v >= 0.40 && v < 0.60) {
            value_ = "";
        }
        else {
            value_ = "Light";
        }
        // Low saturation : achromatic
        if (s < 0.15) return (v > 0.75) ? "White" : "Gray";
        // Chromatic: map hue to colour name (hue is 0-179 in OpenCV HSV)
        if (h < 0.069) return value_+" "+"Red";          //   0 – 12
        if (h < 0.208) return value_ + " " + "Orange";       //  12 – 37
        if (h < 0.319) return value_ + " " + "Yellow";       //  37 – 57
        if (h < 0.528) return value_ + " " + "Green";        //  57 – 95
        if (h < 0.681) return value_ + " " + "Cyan";         //  95 – 122
        if (h < 0.792) return value_ + " " + "Blue";         // 122 – 142
        if (h < 0.903) return value_ + " " + "Purple";       // 142 – 162
        if (h < 0.972) return value_ + " " + "Magenta";      // 162 – 174
        return value_ + "Red";                          // 174 – 179 wraps back to red
    }
};