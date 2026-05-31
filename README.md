## About this project
This project was made during my mandatory internship at focus GmbH while studying at HTWK Leipzig.

The cpp code is a wrapper for onnx exported yolov8-seg-n model that was trained to detect individual bricks on a wall but could be applied different use cases. This project runs a tiled inference pipeline with halo tiles for better single objet accuracy. It employs automatical duplication deletion, sorting by criteria of object size or color and conversion from opencv bounding boxes and masks to double pixel coordinates.

### Use
Build the project with cmake, then in terminal:
` yolov8_seg_pt.exe "path-to-your-model.onnx" "path-to-your-image" `
