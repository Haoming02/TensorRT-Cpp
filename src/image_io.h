#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<std::unique_ptr<float[]>>
loadImage(const std::string &imagePath, const int splitDim, const int overlap,
          int &originalWidth, int &originalHeight, int &rows, int &cols);

void saveImage(float **outputData, const std::string &savePath,
               const int mergeDim, const int overlap, const int upscaledWidth,
               const int upscaledHeight, const int rows, const int cols);

#endif
