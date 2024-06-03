#include "config_loader.h"
#include "json.hpp"

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
    config.modelPath = jsonData.at("modelPath").get<string>();
    config.inputResolution = jsonData.at("inputResolution").get<int>();
    config.overlap = jsonData.at("overlap").get<int>();
    config.upscaleRatio = jsonData.at("upscaleRatio").get<int>();
    config.deviceID = jsonData.at("deviceID").get<int>();

    return config;
  } catch (...) {
    cerr << "Failed to parse config..." << endl;
    exit(EXIT_FAILURE);
  }
}
