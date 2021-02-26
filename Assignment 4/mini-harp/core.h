#pragma once

#include "decode.h"
#include "mem.h"
#include "warp.h"

namespace Harp {

class Core {
 public:
  Core(unsigned num_warps, unsigned num_threads, unsigned num_regs,
       const char *output);
  ~Core();

  void load(const char *program);

  bool running() const;

  void step();

  void printStats();

  Decoder *decoder() { return decoder_; }

  Cache *icache() { return icache_; }

  Cache *dcache() { return dcache_; }

  std::vector<Warp *> &warps() { return warps_; }

 private:
  uint64_t getInstructionCount() const;

  RamMemDevice *ram_;
  ConsoleDevice *console_;
  Decoder *decoder_;
  MemoryUnit *mem_;
  Cache *icache_;
  Cache *dcache_;
  std::vector<Warp *> warps_;

  friend class Warp;

  unsigned totalWarps;
};
}  // namespace Harp
