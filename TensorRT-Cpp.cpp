#include "src/config_loader.h"
#include "src/image_io.h"
#include "src/trt_helper.h"

#include "src/caption.h"
#include "src/upscale.h"

using namespace std;

static void trim(string &str) {
  if (str.front() == '"' && str.back() == '"')
    str = str.substr(1, str.size() - 2);
}

int main(int argc, char *argv[]) {
  if (argc > 3) {
    cerr << "Invalid number of arguments..." << endl;
    cout << "Usage:\nTensorRT-Cpp \"<path to image(s)>\" \"<path to config>\""
         << endl;
    exit(EXIT_FAILURE);
  }

  const string configPath =
      (argc == 3) ? argv[2] : combinePaths(getExeDir(), "config.json");
  Config config = parseConfig(configPath);

  if (config.mode == "upscale") {
    cout << "[Upscale Config]" << endl;
    cout << "Overlap:\t" << *(config.overlap) << endl;
    cout << "Upscale Ratio:\t" << *(config.upscaleRatio) << endl;
  } else {
    cout << "[Caption Config]" << endl;
    cout << "Tags Path:\t" << *(config.tagsPath) << endl;
    cout << "Threshold:\t" << *(config.threshold) << endl;
  }

  string imagePath;
  if (argc > 1)
    imagePath = argv[1];
  else {
    cout << "Path to Image(s): ";
    getline(cin, imagePath);
  }

  trim(imagePath);
  vector<Image> inputs = loadImage(imagePath);
  cout << "Loaded " << inputs.size() << " Images... " << endl;

  cudaSetDevice(config.deviceID);

  size_t engineSize;
  unique_ptr<char[]> engineData = loadEngine(config.modelPath, engineSize);

  nvinfer1::IRuntime *runtime;
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;

  createContext(move(engineData), engineSize, runtime, engine, context);

  if (config.mode == "upscale") {
    vector<Image> outputs = processUpscale(config, inputs, context);
    saveImage(outputs, 'x' + to_string(*(config.upscaleRatio)));
  } else {
    vector<Caption> outputs = processCaption(config, inputs, context);
    saveCaption(outputs, ".txt");
  }

  delete context;
  delete engine;
  delete runtime;

  return 0;
}
