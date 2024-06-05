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
    config.tagsPath = jsonData.at("tagsPath").get<string>();
    config.width = jsonData.at("width").get<int>();
    config.height = jsonData.at("height").get<int>();
    config.threshold = jsonData.at("threshold").get<float>();
    config.deviceID = jsonData.at("deviceID").get<int>();

    return config;
  } catch (...) {
    cerr << "Failed to parse config..." << endl;
    exit(EXIT_FAILURE);
  }
}
