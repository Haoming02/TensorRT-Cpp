#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <vector>

template <typename precision>
std::unique_ptr<precision[]> image2floatCHW(const cv::Mat &image,
                                            const int dim);

template <typename precision>
std::unique_ptr<precision[]> image2floatHWC(const cv::Mat &image,
                                            const int dim);

template <typename precision>
cv::Mat float2image(const std::unique_ptr<precision[]> &outputData,
                    const int dim);

std::vector<cv::Mat> splitImage(const cv::Mat &image, const int inputResolution,
                                const int overlap, int &rows, int &cols);

cv::Mat mergeImage(const std::vector<cv::Mat> &subImages,
                   const int upscaledSize, const int overlap, const int rows,
                   const int cols);

#endif
