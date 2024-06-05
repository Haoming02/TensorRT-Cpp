#include "image_io.h"

using namespace std;
using namespace cv;

unique_ptr<float[]> loadImage(const string &imagePath, const int height,
                              const int width) {
  Mat image = imread(imagePath, IMREAD_COLOR);

  if (image.empty()) {
    cerr << "Failed to open image..." << endl;
    exit(EXIT_FAILURE);
  }

  resize(image, image, Size(height, width));

  vector<Mat> channels(3);
  split(image, channels);

  unique_ptr<float[]> result = make_unique<float[]>(3 * height * width);

  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c)
        result[y * width * 3 + x * 3 + c] =
            static_cast<float>(channels[c].at<unsigned char>(y, x));

  return result;
}
