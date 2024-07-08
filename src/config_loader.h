#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <fstream>
#include <iostream>
#include <string>

struct Config {
  int deviceID;
  std::string mode;
  std::string modelPath;
  int inputResolution;
  bool fp16;

  std::unique_ptr<int> overlap;
  std::unique_ptr<int> upscaleRatio;

  std::unique_ptr<std::string> tagsPath;
  std::unique_ptr<float> threshold;
};

Config parseConfig(const std::string &configPath);

#endif
