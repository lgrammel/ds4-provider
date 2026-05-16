#pragma once

#include <cstddef>
#include <string>
#include <vector>

extern "C" {
#include "ds4.h"
}

static ds4_backend ParseBackend(const std::string &backend) {
  if (backend == "cpu") {
    return DS4_BACKEND_CPU;
  }
  if (backend == "cuda") {
    return DS4_BACKEND_CUDA;
  }
#ifdef __APPLE__
  return DS4_BACKEND_METAL;
#else
  return DS4_BACKEND_CPU;
#endif
}

static bool EndsWithStopSequence(const std::string &text,
                                 const std::vector<std::string> &stop_sequences,
                                 size_t *stop_start) {
  for (const auto &stop : stop_sequences) {
    if (stop.empty() || text.size() < stop.size()) {
      continue;
    }
    if (text.compare(text.size() - stop.size(), stop.size(), stop) == 0) {
      *stop_start = text.size() - stop.size();
      return true;
    }
  }
  return false;
}

static ds4_think_mode ParseThinkMode(const std::string &think_mode) {
  if (think_mode == "high") {
    return DS4_THINK_HIGH;
  }
  if (think_mode == "max") {
    return DS4_THINK_MAX;
  }
  return DS4_THINK_NONE;
}
