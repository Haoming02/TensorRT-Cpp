#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>

std::unique_ptr<float[]> image2floatCHW(const cv::Mat &image, const int dim);
std::unique_ptr<float[]> image2floatHWC(const cv::Mat &image, const int dim);

cv::Mat float2image(const std::unique_ptr<float[]> &outputData, const int dim);

std::vector<cv::Mat> splitImage(const cv::Mat &image, const int inputResolution,
                                const int overlap, int &rows, int &cols);

cv::Mat mergeImage(const std::vector<cv::Mat> &subImages,
                   const int upscaledSize, const int overlap, const int rows,
                   const int cols);

#endif
