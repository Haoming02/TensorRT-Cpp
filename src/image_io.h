#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "path_util.h"

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

struct Image {
  cv::Mat mat;
  std::string path;

  Image(const cv::Mat &m, const std::string &p) : mat(m), path(p){};
};

struct Caption {
  std::string tags;
  std::string path;

  Caption(const std::string &t, const std::string &p) : tags(t), path(p){};
};

std::vector<Image> loadImage(const std::string &path);

void saveImage(const std::vector<Image> &images, const std::string &suffix);

void saveCaption(const std::vector<Caption> &captions, const std::string &ext);

#endif
