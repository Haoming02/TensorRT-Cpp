#include "image_processing.h"

using namespace std;
using namespace cv;

unique_ptr<float[]> image2float(const Mat &image, const int dim) {
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

Mat float2image(const float *outputData, const int dim) {
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

static Mat _blendImages(const Mat &img1, const Mat &img2, const float alpha) {
  Mat blended;
  addWeighted(img1, alpha, img2, 1 - alpha, 0, blended);
  return blended;
}

Mat mergeImage(const vector<Mat> &subImages, const int mergeSize,
               const int overlap, const int upscaledWidth,
               const int upscaledHeight, const int rows, const int cols) {
  Mat mergedImage = Mat::zeros(rows * mergeSize, cols * mergeSize, CV_32FC3);

  int idx = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      int startX = x * (mergeSize - overlap);
      int startY = y * (mergeSize - overlap);

      Mat roi = mergedImage(Rect(startX, startY, mergeSize, mergeSize));
      Mat currentSubImage = subImages[idx++];

      if (x > 0) {
        Mat leftOverlap = roi(Rect(0, 0, overlap, mergeSize));
        Mat subImageLeftOverlap =
            currentSubImage(Rect(0, 0, overlap, mergeSize));
        for (int i = 0; i < overlap; ++i) {
          float alpha = float(i) / overlap;
          leftOverlap.col(i) = _blendImages(leftOverlap.col(i),
                                            subImageLeftOverlap.col(i), alpha);
        }
      }

      if (y > 0) {
        Mat topOverlap = roi(Rect(0, 0, mergeSize, overlap));
        Mat subImageTopOverlap =
            currentSubImage(Rect(0, 0, mergeSize, overlap));
        for (int i = 0; i < overlap; ++i) {
          float alpha = float(i) / overlap;
          topOverlap.row(i) =
              _blendImages(topOverlap.row(i), subImageTopOverlap.row(i), alpha);
        }
      }

      if (x > 0 && y > 0) {
        Mat cornerOverlap = roi(Rect(0, 0, overlap, overlap));
        Mat subImageCornerOverlap =
            currentSubImage(Rect(0, 0, overlap, overlap));
        for (int i = 0; i < overlap; ++i) {
          for (int j = 0; j < overlap; ++j) {
            float alpha = float(i * j) / (overlap * overlap);
            cornerOverlap.at<Vec3b>(i, j) =
                cornerOverlap.at<Vec3b>(i, j) * alpha +
                subImageCornerOverlap.at<Vec3b>(i, j) * (1.0f - alpha);
          }
        }
      }

      currentSubImage.copyTo(roi);
    }
  }

  return mergedImage(Rect(0, 0, upscaledWidth, upscaledHeight));
}
