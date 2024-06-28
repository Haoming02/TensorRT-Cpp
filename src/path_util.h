#ifndef PATH_UTIL_H
#define PATH_UTIL_H

#include <string>

std::string combinePaths(const std::string &path1, const std::string &path2);

std::string addSuffix(const std::string &filepath, const std::string &suffix);

std::string addExtension(const std::string &filepath, const std::string &ext);

std::string getExeDir();

#endif
