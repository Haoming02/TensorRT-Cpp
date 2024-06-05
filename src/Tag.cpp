#include "config_loader.h"
#include "image_io.h"
#include "path_util.h"
#include "rapidcsv.h"
#include "trt_helper.h"

using namespace std;

static string addExtension(const string &filepath, const string &ext) {
  const size_t dotPosition = filepath.find_last_of('.');
  if (dotPosition == string::npos)
    return filepath + ext;
  else
    return filepath.substr(0, dotPosition) + ext;
}

static bool sortByFloat(const pair<string, float> &lhs,
                        const pair<string, float> &rhs) {
  return lhs.second > rhs.second;
}

static vector<float> process(const Config &config, const string &imagePath,
                             const int tagCount,
                             nvinfer1::IExecutionContext *&context) {
  unique_ptr<float[]> imageData =
      loadImage(imagePath, config.height, config.width);

  const unsigned int inputSize = 1 * config.width * config.height * 3;

  void *buffers[2];
  cudaMalloc(&buffers[0], inputSize * sizeof(float));
  cudaMalloc(&buffers[1], tagCount * sizeof(float));

  vector<float> outputData(tagCount);

  cudaMemcpy(buffers[0], imageData.get(), inputSize * sizeof(float),
             cudaMemcpyHostToDevice);

  context->executeV2(buffers);

  cudaMemcpy(outputData.data(), buffers[1], tagCount * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);

  return outputData;
}

int main(int argc, char *argv[]) {
  if (argc > 3) {
    cerr << "Invalid number of arguments...\nRemember to add quotation marks "
            "to path with spaces!\n"
         << endl;
    cerr << "Usage:\Tag.exe \"<path to image>\" \"<path to config>\"" << endl;
    exit(EXIT_FAILURE);
  }

  const string configPath =
      (argc < 3) ? combinePaths(getExecutableDirectory(), "config.json")
                 : argv[2];
  const Config config = parseConfig(configPath);

  cudaSetDevice(config.deviceID);

  rapidcsv::Document tagCSV(config.tagsPath);

  const size_t tagCount = tagCSV.GetRowCount();

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

  vector<float> data = process(config, imagePath, tagCount, context);
  const string name = "name";
  const string category = "category";
  const int general = 0;

  vector<pair<string, float>> pairs;

  for (int i = 0; i < tagCount; ++i)
    if (data[i] > config.threshold &&
        tagCSV.GetCell<int>(category, i) == general)
      pairs.push_back(make_pair(tagCSV.GetCell<string>(name, i), data[i]));

  sort(pairs.begin(), pairs.end(), sortByFloat);

  string result;

  for (const auto &pair : pairs)
    result += pair.first + ", ";

  const string outputPath = addExtension(imagePath, ".txt");

  ofstream myfile;
  myfile.open(outputPath);
  myfile << result.substr(0, result.length() - 2);
  myfile.close();

  delete context;
  delete engine;
  delete runtime;

  return 0;
}
