#include "config_loader.h"
#include "../3rd-party/json.hpp"

using namespace std;
using namespace nlohmann;

Config parseConfig(const string &configPath) {
  ifstream configFile(configPath);

  if (!configFile) {
    cerr << "Failed to open config..." << endl;
    exit(EXIT_FAILURE);
  }

  try {
    json jsonData;
    configFile >> jsonData;

    Config config;
    config.deviceID = jsonData.at("deviceID").get<int>();
    config.mode = jsonData.at("mode").get<string>();
    config.modelPath = jsonData.at("modelPath").get<string>();
    config.inputResolution = jsonData.at("inputResolution").get<int>();
    config.fp16 = jsonData.at("fp16").get<bool>();

    if (config.mode == "upscale") {
      config.overlap = make_unique<int>(jsonData.at("overlap").get<int>());
      config.upscaleRatio =
          make_unique<int>(jsonData.at("upscaleRatio").get<int>());

      config.tagsPath = nullptr;
      config.threshold = nullptr;
    } else if (config.mode == "caption") {
      config.overlap = nullptr;
      config.upscaleRatio = nullptr;

      config.tagsPath =
          make_unique<string>(jsonData.at("tagsPath").get<string>());
      config.threshold =
          make_unique<float>(jsonData.at("threshold").get<float>());
    } else {
      cerr << "Unrecognized Mode: \"" << config.mode << "\"... " << endl;
      exit(EXIT_FAILURE);
    }

    return config;
  } catch (...) {
    cerr << "Failed to parse config..." << endl;
    exit(EXIT_FAILURE);
  }
}
