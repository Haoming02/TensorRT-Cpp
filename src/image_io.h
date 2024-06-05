#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <opencv2/opencv.hpp>
#include <vector>

std::unique_ptr<float[]> loadImage(const std::string &imagePath,
                                   const int height, const int width);

#endif
