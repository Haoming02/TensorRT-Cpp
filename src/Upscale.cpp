#include "config_loader.h"
#include "image_io.h"
#include "path_util.h"
#include "trt_helper.h"

#include <cmath>
#include <string>

using namespace std;

static string addSuffix(const string &filepath, const string &suffix) {
  const size_t dotPosition = filepath.find_last_of('.');
  if (dotPosition == string::npos)
    return filepath + suffix;
  else
    return filepath.substr(0, dotPosition) + suffix +
           filepath.substr(dotPosition);
}

static void process(const Config &config, const string &imagePath,
                    nvinfer1::IExecutionContext *&context) {
  int og_w, og_h, rows, cols;
  vector<unique_ptr<float[]>> imageData =
      loadImage(imagePath, config.inputResolution, config.overlap, og_w, og_h,
                rows, cols);

  const unsigned int inputSize = 1 * 3 * pow(config.inputResolution, 2);
  const unsigned int outputSize = inputSize * pow(config.upscaleRatio, 2);

  void *buffers[2];
  cudaMalloc(&buffers[0], inputSize * sizeof(float));
  cudaMalloc(&buffers[1], outputSize * sizeof(float));

  const int subImageCount = imageData.size();
  float **outputData = new float *[subImageCount];

  for (int i = 0; i < subImageCount; ++i) {
    cudaMemcpy(buffers[0], imageData[i].get(), inputSize * sizeof(float),
               cudaMemcpyHostToDevice);

    context->executeV2(buffers);
    outputData[i] = new float[outputSize];

    cudaMemcpy(outputData[i], buffers[1], outputSize * sizeof(float),
               cudaMemcpyDeviceToHost);
  }

  const string resultPath =
      addSuffix(imagePath, to_string(config.upscaleRatio) + "x");

  saveImage(outputData, resultPath,
            config.inputResolution * config.upscaleRatio,
            config.overlap * config.upscaleRatio, og_w * config.upscaleRatio,
            og_h * config.upscaleRatio, rows, cols);

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);

  for (int i = 0; i < subImageCount; ++i)
    delete[] outputData[i];

  delete[] outputData;
}

int main(int argc, char *argv[]) {
  if (argc > 3) {
    cerr << "Invalid number of arguments...\nRemember to add quotation marks "
            "to path with spaces!\n"
         << endl;
    cerr << "Usage:\nUpscale.exe \"<path to image>\" \"<path to config>\""
         << endl;
    exit(EXIT_FAILURE);
  }

  const string configPath =
      (argc < 3) ? combinePaths(getExecutableDirectory(), "config.json")
                 : argv[2];
  const Config config = parseConfig(configPath);

  cudaSetDevice(config.deviceID);

  size_t engineSize;
  unique_ptr<char[]> engineData = loadEngine(config.modelPath, engineSize);

  nvinfer1::IRuntime *runtime;
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;

  createContext(move(engineData), engineSize, runtime, engine, context);

  string imagePath;
  if (argc >= 2)
    imagePath = argv[1];
  else {
    cout << "Path to Image: ";
    getline(cin, imagePath);
  }

  process(config, imagePath, context);

  delete context;
  delete engine;
  delete runtime;

  return 0;
}
