/*
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
        return value_ + " " + "Red";                          // 174 – 179 wraps back to red
    }
};
*/

#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <functional>

constexpr double PI = 3.14159265358979323846;
// All points are in CIE Lab space:
//   L* : [0, 100]   (lightness)
//   a* : [-128, 127] (green <-> red)
//   b* : [-128, 127] (blue <-> yellow)

using Point3D = std::array<double, 3>;
using Vector3D = std::array<double, 3>;

struct DetectedColorSubSpace {
    Vector3D bounds;
    int      id;
    std::string assigned_color;
};

class DynamicGrid {

public:
    static constexpr double L_AXIS = 100.0;
    static constexpr double A_AXIS = 255.0;
    static constexpr double B_AXIS = 255.0;

    static constexpr double A_OFFSET = 128.0;
    static constexpr double B_OFFSET = 128.0;

    explicit DynamicGrid(double percent) {
        if (percent <= 0.0 || percent > 100.0)
            throw std::invalid_argument("percent must be in the range (0, 100]");

        divisions_ = static_cast<int>(std::round(100.0 / percent));

        cell_size_[0] = L_AXIS / divisions_;
        cell_size_[1] = A_AXIS / divisions_;
        cell_size_[2] = B_AXIS / divisions_;

        origin_ = { 0.0, 0.0, 0.0 };
    }

    // Expects point in raw Lab: L*[0,100], a*[-128,127], b*[-128,127]
    void isInBoundsOfSubspace(const Point3D& point, int subspaceID) const {
        const int total = divisions_ * divisions_ * divisions_;
        if (subspaceID < 0 || subspaceID >= total)
            throw std::out_of_range("subspaceID " + std::to_string(subspaceID) +
                " is out of range [0, " + std::to_string(total - 1) + "]");

        const Point3D shifted = toShifted(point);

        const int li = subspaceID / (divisions_ * divisions_);
        const int ai = (subspaceID / divisions_) % divisions_;
        const int bi = subspaceID % divisions_;

        const double l_min = origin_[0] + li * cell_size_[0];
        const double a_min = origin_[1] + ai * cell_size_[1];
        const double b_min = origin_[2] + bi * cell_size_[2];

        const bool inside =
            (shifted[0] >= l_min && shifted[0] < l_min + cell_size_[0]) &&
            (shifted[1] >= a_min && shifted[1] < a_min + cell_size_[1]) &&
            (shifted[2] >= b_min && shifted[2] < b_min + cell_size_[2]);

        if (!inside)
            throw std::domain_error(
                "Point (" + std::to_string(point[0]) + ", " +
                std::to_string(point[1]) + ", " +
                std::to_string(point[2]) +
                ") is not within subspace " + std::to_string(subspaceID));
    }

    [[nodiscard]]
    DetectedColorSubSpace getDCSS(Point3D point) {

        point[0] = std::clamp(point[0], 0.0, L_AXIS);
        point[1] = std::clamp(point[1], -128.0, 127.0);
        point[2] = std::clamp(point[2], -128.0, 127.0);

        const Point3D shifted = toShifted(point);

        const int li = std::min(static_cast<int>(shifted[0] / cell_size_[0]), divisions_ - 1);
        const int ai = std::min(static_cast<int>(shifted[1] / cell_size_[1]), divisions_ - 1);
        const int bi = std::min(static_cast<int>(shifted[2] / cell_size_[2]), divisions_ - 1);

        const int id = li * divisions_ * divisions_ + ai * divisions_ + bi;

        DetectedColorSubSpace dcss;
        dcss.bounds = cell_size_;
        dcss.id = id;
        dcss.assigned_color = labCellToColorName(li, ai, bi);
        return dcss;
    }

private:
    int      divisions_;
    Vector3D cell_size_;
    Point3D  origin_;

    [[nodiscard]]
    static Point3D toShifted(const Point3D& p) {
        return { p[0], p[1] + A_OFFSET, p[2] + B_OFFSET };
    }

    [[nodiscard]]
    std::string labCellToColorName(int li, int ai, int bi) const {
        // Normalise cell centres to [0,1]
        const double L = ((li + 0.5) * cell_size_[0]) / L_AXIS;
        const double A = ((ai + 0.5) * cell_size_[1]) / A_AXIS;
        const double B = ((bi + 0.5) * cell_size_[2]) / B_AXIS;

        // --- Lightness tier ---
        if (L < 0.20) return "Black";
        if (L > 0.85) return "White";

        const std::string lightness = (L < 0.40) ? "Dark" : (L < 0.65) ? "" : "Light";

        const double da = A - 0.5;
        const double db = B - 0.5;

        const double chroma = std::sqrt(da * da + db * db); // 0-0.707

        if (chroma < 0.10) {
            return (L < 0.40) ? "Dark Gray" : "Gray";
        }

        const double hue = std::fmod(std::atan2(db, da) * (180.0 / PI) + 360.0, 360.0);

        std::string color;
        if (hue < 30.0) color = "Red";
        else if (hue < 60.0) color = "Orange";
        else if (hue < 90.0) color = "Yellow";
        else if (hue < 150.0) color = "Green";
        else if (hue < 210.0) color = "Cyan";
        else if (hue < 270.0) color = "Blue";
        else if (hue < 330.0) color = "Purple";
        else                  color = "Red";   // wraps back

        return lightness.empty() ? color : lightness + " " + color;
    }
};

class DBSCAN {
public:
    static constexpr int NOISE = -1;
    static constexpr int UNVISITED = -2;

    // eps     : maximum ΔE distance to be considered a neighbour
    // minPts  : minimum neighbours to form a core point
    // distFn  : distance function (defaults to Euclidean in Lab)
    DBSCAN(double eps, int minPts,
        std::function<double(const Point3D&, const Point3D&)> distFn = euclidean)
        : eps_(eps), minPts_(minPts), distFn_(distFn) {}

    // Returns a label per point:
    //   >= 0  : cluster ID
    //   NOISE : does not belong to any cluster
    std::vector<int> fit(const std::vector<Point3D>& points) {
        const int n = static_cast<int>(points.size());
        std::vector<int> labels(n, UNVISITED);
        int clusterID = 0;

        for (int i = 0; i < n; i++) {
            if (labels[i] != UNVISITED) continue;

            std::vector<int> neighbours = regionQuery(points, i);

            if (static_cast<int>(neighbours.size()) <= minPts_) {
                labels[i] = NOISE;
                std::cout << "label" << i << " ist NOISE zugewiesen";
                continue;
            }

            expandCluster(points, labels, i, neighbours, clusterID);
            clusterID++;
        }
        return labels;
    }

    static double euclidean(const Point3D& a, const Point3D& b) {
        double dL = a[0] - b[0];
        double da = a[1] - b[1];
        double db = a[2] - b[2];
        return std::sqrt(dL * dL + da * da + db * db);
    }

private:
    double   eps_;
    int      minPts_;
    std::function<double(const Point3D&, const Point3D&)> distFn_;

    std::vector<int> regionQuery(const std::vector<Point3D>& points, int idx) const {
        std::vector<int> neighbours;
        for (int i = 0; i < static_cast<int>(points.size()); i++)
            if (i != idx && distFn_(points[idx], points[i]) <= eps_)
                neighbours.push_back(i);
        return neighbours;
    }

    void expandCluster(const std::vector<Point3D>& points, std::vector<int>& labels,
        int idx, std::vector<int> neighbours, int clusterID) {
        labels[idx] = clusterID;

        for (int i = 0; i < static_cast<int>(neighbours.size()); i++) {
            int nb = neighbours[i];

            if (labels[nb] == NOISE) { labels[nb] = clusterID; continue; }
            if (labels[nb] != UNVISITED) continue;

            labels[nb] = clusterID;

            std::vector<int> nbNeighbours = regionQuery(points, nb);
            if (static_cast<int>(nbNeighbours.size()) >= minPts_)
                neighbours.insert(neighbours.end(), nbNeighbours.begin(), nbNeighbours.end());
        }
    }
};