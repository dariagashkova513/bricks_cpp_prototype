cmake_minimum_required(VERSION 3.18)
project(yolov8_seg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of ONNX Runtime install")

if(ONNXRUNTIME_ROOT)
    set(ORT_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    find_library(ORT_LIBS NAMES onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
else()
    find_path(ORT_INCLUDE_DIRS NAMES onnxruntime/core/session/onnxruntime_cxx_api.h REQUIRED)
    find_library(ORT_LIBS NAMES onnxruntime REQUIRED)
endif()

# --- detect-seg (instance segmentation) ---
add_executable(yolov8_seg main_seg.cpp yolov8seg.cpp)
target_include_directories(yolov8_seg PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
target_link_libraries(yolov8_seg PRIVATE ${OpenCV_LIBS} ${ORT_LIBS})