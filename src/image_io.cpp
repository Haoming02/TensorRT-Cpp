#include "image_io.h"
#include "image_processing.h"

using namespace std;
using namespace cv;

vector<unique_ptr<float[]>> loadImage(const string &imagePath,
                                      const int splitDim, const int overlap,
                                      int &originalWidth, int &originalHeight,
                                      int &rows, int &cols) {
  Mat image = imread(imagePath, IMREAD_COLOR);

  if (image.empty()) {
    cerr << "Failed to open image..." << endl;
    exit(EXIT_FAILURE);
  } else {
    cout << "Loaded Image of Size: ";
    cout << image.size << endl;
    originalWidth = image.size().width;
    originalHeight = image.size().height;
  }

  vector<Mat> subImages = splitImage(image, splitDim, overlap, rows, cols);

  vector<unique_ptr<float[]>> imageData;

  for (const Mat &subimage : subImages)
    imageData.push_back(image2float(subimage, splitDim));

  return imageData;
}

void saveImage(float **imageData, const std::string &savePath,
               const int mergeDim, const int overlap, const int upscaledWidth,
               const int upscaledHeight, const int rows, const int cols) {
  vector<Mat> subImages;

  for (int i = 0; i < rows * cols; ++i)
    subImages.push_back(float2image(imageData[i], mergeDim));

  Mat combinedImage = mergeImage(subImages, mergeDim, overlap, upscaledWidth,
                                 upscaledHeight, rows, cols);

  combinedImage *= 255.0;
  combinedImage.convertTo(combinedImage, CV_8UC3);

  imwrite(savePath, combinedImage);
}
