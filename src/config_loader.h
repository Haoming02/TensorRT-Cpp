#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <fstream>
#include <iostream>
#include <string>

struct Config {
  std::string modelPath;
  int inputResolution;
  int overlap;
  int upscaleRatio;
  int deviceID;
};

Config parseConfig(const std::string &configPath);

#endif
