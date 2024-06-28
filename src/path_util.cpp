#include "path_util.h"

std::string combinePaths(const std::string &path1, const std::string &path2) {
  if (path1.empty())
    return path2;
  if (path2.empty())
    return path1;
  if (path1.back() == '\\' || path1.back() == '/')
    return path1 + path2;
  return path1 + '\\' + path2;
}

std::string addSuffix(const std::string &filepath, const std::string &suffix) {
  const size_t dotPosition = filepath.find_last_of('.');

  if (dotPosition == std::string::npos)
    return filepath + suffix;
  else
    return filepath.substr(0, dotPosition) + suffix +
           filepath.substr(dotPosition);
}

std::string addExtension(const std::string &filepath, const std::string &ext) {
  const size_t dotPosition = filepath.find_last_of('.');

  if (dotPosition == std::string::npos)
    return filepath + ext;
  else
    return filepath.substr(0, dotPosition) + ext;
}

// ========================== //
// PLATFORM SPECIFIC: WINDOWS //
// ========================== //

#include <windows.h>

std::string getExeDir() {
  char buffer[MAX_PATH];
  GetModuleFileNameA(NULL, buffer, MAX_PATH);
  size_t pos = std::string(buffer).find_last_of("\\/");
  return std::string(buffer).substr(0, pos);
}
