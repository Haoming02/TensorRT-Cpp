#include "caption.h"
#include "../3rd-party/rapidcsv.h"
#include "image_processing.h"

using namespace std;

struct Data {
  string tag;
  float weight;

  Data(const string &t, const float &w) : tag(t), weight(w){};
};

static bool sortByWeight(const Data &lhs, const Data &rhs) {
  return lhs.weight > rhs.weight;
}

vector<Caption> processCaption(const Config &config,
                               const vector<Image> &inputs,
                               nvinfer1::IExecutionContext *&context) {
  vector<Caption> results;

  rapidcsv::Document tagCSV(*(config.tagsPath));
  const string name = "name";
  const string category = "category";
  const int general = 0;

  const size_t tagCount = tagCSV.GetRowCount();
  const int inputSize = 1 * pow(config.inputResolution, 2) * 3;

  void *buffers[2];
  cudaMalloc(&buffers[0], inputSize * sizeof(float));
  cudaMalloc(&buffers[1], tagCount * sizeof(float));

  cv::Mat input;
  vector<float> outputData(tagCount);

  for (const auto &image : inputs) {
    cv::resize(image.mat, input,
               cv::Size(config.inputResolution, config.inputResolution));

    unique_ptr<float[]> imageData =
        image2floatHWC(input, config.inputResolution);

    cudaMemcpy(buffers[0], imageData.get(), inputSize * sizeof(float),
               cudaMemcpyHostToDevice);

    context->executeV2(buffers);

    cudaMemcpy(outputData.data(), buffers[1], tagCount * sizeof(float),
               cudaMemcpyDeviceToHost);

    vector<Data> pairs;

    for (int i = 0; i < tagCount; ++i) {
      if (outputData[i] > *(config.threshold) &&
          tagCSV.GetCell<int>(category, i) == general)
        pairs.push_back(Data(tagCSV.GetCell<string>(name, i), outputData[i]));
    }

    sort(pairs.begin(), pairs.end(), sortByWeight);

    string caption = pairs[0].tag;
    pairs.erase(pairs.begin());
    for (const auto &pair : pairs)
      caption += ", " + pair.tag;

    results.push_back(Caption(caption, image.path));
  }

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);

  return results;
}
