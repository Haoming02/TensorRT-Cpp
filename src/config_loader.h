#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <fstream>
#include <iostream>
#include <string>

struct Config {
  std::string modelPath;
  std::string tagsPath;
  int width;
  int height;
  float threshold;
  int deviceID;
};

Config parseConfig(const std::string &configPath);

#endif
