cmake_minimum_required(VERSION 3.18)
project(yolov8_seg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV ---
# Point to the directory that actually contains OpenCVConfig.cmake
find_package(OpenCV REQUIRED
    PATHS "C:/Users/Supervisor/libs/opencv/build/x64/vc16"
    NO_DEFAULT_PATH
)

# --- ONNX Runtime ---
# Expect these to be passed from presets or use fallback PATHS
find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "C:/Users/Supervisor/libs/onnxruntime-win-x64-1.24.4/include"
    REQUIRED
)

find_library(ORT_LIBRARY
    NAMES onnxruntime
    PATHS "C:/Users/Supervisor/libs/onnxruntime-win-x64-1.24.4/lib"
    REQUIRED
)

# --- Executable ---
add_executable(yolov8_seg
    main.cpp
    yolov8_seg_pt.cpp
)

# --- Includes ---
target_include_directories(yolov8_seg PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
)

# --- Libraries ---
target_link_libraries(yolov8_seg PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIBRARY}
)
