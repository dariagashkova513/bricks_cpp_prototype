cmake_minimum_required(VERSION 3.18)
project(yolov8_onnx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/libs_cpp/opencv-4.11.0/build") 
find_package(OpenCV REQUIRED)
include_directories("C:/libs_cpp/opencv-4.11.0/include")

set(ONNXRUNTIME_ROOT "C:/libs_cpp/onnxruntime-win-x64-1.24.4")

if(ONNXRUNTIME_ROOT)
    set(ORT_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    find_library(ORT_LIBS NAMES onnxruntime
                 HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
else()
    find_path(ORT_INCLUDE_DIRS
              NAMES onnxruntime/core/session/onnxruntime_cxx_api.h REQUIRED)
    find_library(ORT_LIBS NAMES onnxruntime REQUIRED)
endif()

# Segmentation
add_executable(yolov8_seg main.cpp yolov8_seg.cpp)
target_include_directories(yolov8_seg PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
target_link_libraries(yolov8_seg PRIVATE ${OpenCV_LIBS} ${ORT_LIBS})
