#include "image_processing.h"

using namespace std;
using namespace cv;

unique_ptr<float[]> image2floatCHW(const Mat &image, const int dim) {
  vector<Mat> channels(3);
  split(image, channels);

  unique_ptr<float[]> result = make_unique<float[]>(3 * dim * dim);

  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < dim; ++y)
      for (int x = 0; x < dim; ++x)
        result[c * dim * dim + y * dim + x] =
            static_cast<float>(channels[2 - c].at<unsigned char>(y, x)) /
            255.0f;

  return result;
}

unique_ptr<float[]> image2floatHWC(const Mat &image, const int dim) {
  vector<Mat> channels(3);
  split(image, channels);

  unique_ptr<float[]> result = make_unique<float[]>(3 * dim * dim);

  for (int y = 0; y < dim; ++y)
    for (int x = 0; x < dim; ++x)
      for (int c = 0; c < 3; ++c)
        result[y * dim * 3 + x * 3 + c] =
            static_cast<float>(channels[c].at<unsigned char>(y, x));

  return result;
}

Mat float2image(const unique_ptr<float[]> &outputData, const int dim) {
  vector<Mat> channels{Mat(dim, dim, CV_32F), Mat(dim, dim, CV_32F),
                       Mat(dim, dim, CV_32F)};

  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < dim; ++y)
      for (int x = 0; x < dim; ++x)
        channels[2 - c].at<float>(y, x) =
            outputData[c * dim * dim + y * dim + x];

  Mat combinedImage;
  merge(channels, combinedImage);

  return combinedImage;
}

static int _calculatePadding(const int val, const int splitSize,
                             const int overlap, int &count) {
  int target = splitSize;
  count = 1;

  while (val > target) {
    target += splitSize - overlap;
    count++;
  }

  return target - val;
}

vector<Mat> splitImage(const Mat &image, const int splitSize, const int overlap,
                       int &rows, int &cols) {
  const int padX = _calculatePadding(image.cols, splitSize, overlap, rows);
  const int padY = _calculatePadding(image.rows, splitSize, overlap, cols);

  Mat paddedImage;
  const Mat *img;

  if (padX > 0 || padY > 0) {
    copyMakeBorder(image, paddedImage, 0, padY, 0, padX, BORDER_REFLECT);
    img = &paddedImage;
  } else {
    img = &image;
  }

  vector<Mat> subImages;

  int startX = 0;
  int startY = 0;

  const int height = (*img).rows;
  const int width = (*img).cols;

  while (startY + splitSize <= height) {
    startX = 0;

    while (startX + splitSize <= width) {
      Rect roi(startX, startY, splitSize, splitSize);
      subImages.push_back((*img)(roi));

      startX += splitSize - overlap;
    }

    startY += splitSize - overlap;
  }

  return subImages;
}

static void fadeEdges(Mat &image, const int overlap, const bool L, const bool R,
                      const bool T, const bool D, const int dim) {
  for (int y = 0; y < dim; ++y) {
    for (int x = 0; x < dim; ++x) {

      const int distLeft = x;
      const int distRight = (dim - x - 1);
      const int distTop = y;
      const int distBottom = (dim - y - 1);

      float alpha = 1.0f;

      if (L && (distLeft < overlap))
        alpha *= static_cast<float>(distLeft + 1) / overlap;
      if (R && (distRight < overlap))
        alpha *= static_cast<float>(distRight) / overlap;
      if (T && (distTop < overlap))
        alpha *= static_cast<float>(distTop + 1) / overlap;
      if (D && (distBottom < overlap))
        alpha *= static_cast<float>(distBottom) / overlap;

      if (alpha < 1.0f)
        image.at<Vec3f>(y, x) *= alpha;
    }
  }
}

Mat mergeImage(const vector<Mat> &subImages, const int upscaledSize,
               const int overlap, const int rows, const int cols) {
  Mat mergedImage =
      Mat::zeros(rows * upscaledSize, cols * upscaledSize, CV_32FC3);

  int idx = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int startX = x * (upscaledSize - overlap);
      int startY = y * (upscaledSize - overlap);

      Mat roi = mergedImage(Rect(startX, startY, upscaledSize, upscaledSize));
      Mat currentSubImage = subImages[idx++];

      const bool L = (x > 0);
      const bool R = (x < cols - 1);
      const bool T = (y > 0);
      const bool D = (y < rows - 1);

      fadeEdges(currentSubImage, overlap, L, R, T, D, upscaledSize);

      roi += currentSubImage;
    }
  }

  return mergedImage;
}
