#include "image_io.h"

using namespace std;
using namespace cv;

namespace fs = filesystem;

vector<Image> loadImage(const string &path) {
  vector<Image> images;

  if (fs::is_directory(path)) {

    for (const auto &entry : fs::directory_iterator(path)) {
      string filePath = entry.path().string();
      Mat mat = imread(filePath, IMREAD_COLOR);
      if (mat.empty())
        continue;

      images.push_back(Image(mat, filePath));
    }

  } else {
    Mat mat = imread(path, IMREAD_COLOR);

    if (!mat.empty())
      images.push_back(Image(mat, path));
  }

  if (images.size() == 0) {
    cerr << "Failed to read any image..." << endl;
    exit(EXIT_FAILURE);
  }

  return images;
}

void saveImage(const vector<Image> &images, const string &suffix) {
  for (const Image &image : images) {
    imwrite(addSuffix(image.path, suffix), image.mat);
  }
}

void saveCaption(const vector<Caption> &captions, const string &ext) {
  for (const Caption &caption : captions) {
    ofstream file;
    file.open(addExtension(caption.path, ext));
    file << caption.tags;
    file.close();
  }
}
