#pragma once

#include "common.h"

namespace Harp {

class LRUCache {
 public:
  LRUCache(uint64_t capacity, uint32_t blockSize, uint32_t associativity);
  ~LRUCache();

  bool lookUp(uint64_t addr, bool write);

  std::pair<bool, uint64_t> insert(uint64_t addr, bool dirty);

  uint32_t getBlockSize() const { return 1ull << blockBits_; }

  uint32_t getNumSets() const { return 1ull << setBits_; }

  uint32_t getAssociativity() const { return 1ull << assocBits_; }

  uint32_t getCapacity() const {
    return 1ull << (blockBits_ + setBits_ + assocBits_);
  }

  uint32_t getFreeSpace() const { return freeSpace_; }

 protected:
  enum {
    MAX_ASSOC_BITS = 30,
  };

  struct tag_t {
    uint64_t tag;
    uint32_t pos : MAX_ASSOC_BITS;
    uint32_t valid : 1;
    uint32_t dirty : 1;
  };

  uint64_t getTag(uint64_t addr) const {
    return addr >> (blockBits_ + setBits_);
  }

  uint32_t getSet(uint64_t addr) const {
    return (addr >> blockBits_) & ((1 << setBits_) - 1);
  }

  void updatePos(uint32_t set_idx, uint32_t tag_idx);

  uint32_t blockBits_;
  uint32_t setBits_;
  uint32_t assocBits_;
  uint32_t freeSpace_;
  std::vector<tag_t> tags_;
};

}  // namespace Harp
