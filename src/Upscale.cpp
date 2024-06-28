#include "upscale.h"
#include "image_processing.h"

using namespace std;
using namespace cv;

std::vector<Image> processUpscale(const Config &config,
                                  const std::vector<Image> &inputs,
                                  nvinfer1::IExecutionContext *&context) {
  vector<Image> results;

  const int inputSize = 1 * 3 * pow(config.inputResolution, 2);
  const int outputSize = inputSize * pow(*(config.upscaleRatio), 2);

  void *buffers[2];
  cudaMalloc(&buffers[0], inputSize * sizeof(float));
  cudaMalloc(&buffers[1], outputSize * sizeof(float));

  for (const auto &image : inputs) {

    const int og_w = image.mat.cols;
    const int og_h = image.mat.rows;

    int rows, cols;
    vector<Mat> subImages = splitImage(image.mat, config.inputResolution,
                                       *(config.overlap), rows, cols);

    vector<unique_ptr<float[]>> inputData;
    for (const Mat &img : subImages)
      inputData.push_back(image2floatCHW(img, config.inputResolution));

    const int subImageCount = inputData.size();
    vector<unique_ptr<float[]>> outputData(subImageCount);

    for (int i = 0; i < subImageCount; ++i) {
      cudaMemcpy(buffers[0], inputData[i].get(), inputSize * sizeof(float),
                 cudaMemcpyHostToDevice);

      context->executeV2(buffers);
      outputData[i] = make_unique<float[]>(outputSize);

      cudaMemcpy(outputData[i].get(), buffers[1], outputSize * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }

    subImages.clear();

    for (const auto &img : outputData)
      subImages.push_back(
          float2image(img, config.inputResolution * *(config.upscaleRatio)));

    // int id = 0;
    // for (auto img : subImages) {
    //   img *= 255.0;
    //   img.convertTo(img, CV_8UC3);
    //   results.push_back(Image(img, addSuffix(image.path, to_string(id))));
    //   id++;
    // }

    Mat mergedImage =
        mergeImage(subImages, config.inputResolution * *(config.upscaleRatio),
                   *(config.overlap) * *(config.upscaleRatio), rows, cols);

    mergedImage *= 255.0;
    mergedImage.convertTo(mergedImage, CV_8UC3);

    mergedImage = mergedImage(Rect(0, 0, og_w * *(config.upscaleRatio),
                                   og_h * *(config.upscaleRatio)));

    results.push_back(Image(mergedImage, image.path));
  }

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);

  return results;
}
