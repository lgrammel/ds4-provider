#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "../native/binding-helpers.h"

static void TestParseBackend() {
  assert(ParseBackend("cpu") == DS4_BACKEND_CPU);
  assert(ParseBackend("cuda") == DS4_BACKEND_CUDA);

#ifdef __APPLE__
  assert(ParseBackend("") == DS4_BACKEND_METAL);
  assert(ParseBackend("metal") == DS4_BACKEND_METAL);
  assert(ParseBackend("unknown") == DS4_BACKEND_METAL);
#else
  assert(ParseBackend("") == DS4_BACKEND_CPU);
  assert(ParseBackend("metal") == DS4_BACKEND_CPU);
  assert(ParseBackend("unknown") == DS4_BACKEND_CPU);
#endif
}

static void TestParseThinkMode() {
  assert(ParseThinkMode("high") == DS4_THINK_HIGH);
  assert(ParseThinkMode("max") == DS4_THINK_MAX);
  assert(ParseThinkMode("none") == DS4_THINK_NONE);
  assert(ParseThinkMode("") == DS4_THINK_NONE);
  assert(ParseThinkMode("unknown") == DS4_THINK_NONE);
}

static void TestEndsWithStopSequence() {
  size_t stop_start = 999;
  assert(EndsWithStopSequence("hello<stop>", {"<stop>"}, &stop_start));
  assert(stop_start == 5);

  stop_start = 999;
  assert(EndsWithStopSequence("abc", {"", "bc"}, &stop_start));
  assert(stop_start == 1);

  stop_start = 999;
  assert(!EndsWithStopSequence("abc", {"abcd", "ab"}, &stop_start));
  assert(stop_start == 999);
}

int main() {
  TestParseBackend();
  TestParseThinkMode();
  TestEndsWithStopSequence();

  std::cout << "binding helper tests passed\n";
  return 0;
}
