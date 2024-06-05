#include "trt_helper.h"

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity < Severity::kINFO)
      std::cerr << msg << std::endl;
  }
} nvLogger;

static void printVersion() {
  const unsigned int major = NV_TENSORRT_VERSION / 10000L;
  const unsigned int minor = (NV_TENSORRT_VERSION % 10000L) / 100L;
  const unsigned int patch = (NV_TENSORRT_VERSION % 100L) / 1L;
  cout << "TensorRT Version: " << major << "." << minor << "." << patch << endl;
}

unique_ptr<char[]> loadEngine(const string &filename, size_t &engineSize) {
  // printVersion();

  ifstream file(filename, ios::binary | ios::ate);
  if (!file.is_open()) {
    cerr << "Failed to open engine file..." << endl;
    exit(EXIT_FAILURE);
  }

  engineSize = file.tellg();
  file.seekg(0, ios::beg);

  unique_ptr<char[]> buffer = make_unique<char[]>(engineSize);
  if (!file.read(buffer.get(), engineSize)) {
    cerr << "Failed to read engine file..." << endl;
    exit(EXIT_FAILURE);
  }

  // cout << "Loaded Engine of Size: ";
  // cout << engineSize / 1000.0L / 1000.0L << "MB" << endl;
  // cout << "from: " << filename << "\n" << endl;

  return buffer;
}

void createContext(const unique_ptr<char[]> &engineData,
                   const size_t engineSize, IRuntime *&runtime,
                   ICudaEngine *&engine, IExecutionContext *&context) {
  runtime = createInferRuntime(nvLogger);
  if (!runtime) {
    cerr << "Failed to create TensorRT runtime..." << endl;
    exit(EXIT_FAILURE);
  }

  engine = runtime->deserializeCudaEngine(engineData.get(), engineSize);
  if (!engine) {
    cerr << "Failed to deserialize CUDA engine..." << endl;
    exit(EXIT_FAILURE);
  }

  context = engine->createExecutionContext();
  if (!context) {
    cerr << "Failed to create execution context..." << endl;
    exit(EXIT_FAILURE);
  }
}
