#ifndef PATH_UTIL_H
#define PATH_UTIL_H

#include <string>
#include <windows.h>

// ========================== //
// PLATFORM SPECIFIC: WINDOWS //
// ========================== //

std::string getExecutableDirectory() {
  char buffer[MAX_PATH];
  GetModuleFileNameA(NULL, buffer, MAX_PATH);
  std::string::size_type pos = std::string(buffer).find_last_of("\\/");
  return std::string(buffer).substr(0, pos);
}

std::string combinePaths(const std::string &path1, const std::string &path2) {
  if (path1.empty())
    return path2;
  if (path2.empty())
    return path1;
  if (path1.back() == '\\' || path1.back() == '/')
    return path1 + path2;
  return path1 + "\\" + path2;
}

#endif
