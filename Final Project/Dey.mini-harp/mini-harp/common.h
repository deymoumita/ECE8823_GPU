#pragma once

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include "assert.h"

#define ADDR_SIZE 4
#define NUM_WARPS 2
#define NUM_TREADS 4
#define NUM_GREGS 16
#define NUM_PREGS 1

#define CONSOLE_ADDR 0x1ffffc
#define RAM_SIZE 65536
#define PAGE_SIZE 4096

#define ICACHE_SIZE 1024
#define ICACHE_BLOCK 32
#define ICACHE_ASSOC 4

#define DCACHE_SIZE 4096
#define DCACHE_BLOCK 32
#define DCACHE_ASSOC 4

typedef uint8_t byte_t;
typedef uint32_t word_t;
typedef int32_t sword_t;
typedef uint32_t addr_t;

#ifdef USE_DEBUG
#define D(lvl, x)                                                              \
  do {                                                                         \
    if ((lvl) <= USE_DEBUG) {                                                  \
      std::cerr << std::endl << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": " \
                << x;                                             \
    }                                                                          \
  } while (0)

#define D_RAW(x)    \
  do {              \
    std::cerr << x; \
  } while (0)
#else
#define D(lvl, x) \
  do {            \
  } while (0)
#endif
