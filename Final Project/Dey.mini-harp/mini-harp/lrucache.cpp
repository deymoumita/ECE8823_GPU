#include "lrucache.h"
#include "utils.h"

using namespace Harp;

LRUCache::LRUCache(uint64_t capacity, uint32_t blockSize,
                   uint32_t associativity) {
  blockBits_ = ceilLog2(blockSize);
  auto num_blocks = capacity >> blockBits_;
  auto num_sets = num_blocks / associativity;
  setBits_ = ceilLog2(num_sets);
  assocBits_ = ceilLog2(associativity);
  assert(assocBits_ <= MAX_ASSOC_BITS);
  assert(capacity == this->getCapacity());

  // initialize tag buffer
  tags_.resize(num_sets * associativity);
  for (uint32_t s = 0; s < num_sets; ++s) {
    for (uint32_t a = 0; a < associativity; ++a) {
      auto i = s * associativity + a;
      tags_[i].pos = a;
      tags_[i].valid = 0;
      tags_[i].dirty = 0;
    }
  }
  freeSpace_ = num_blocks;
}

LRUCache::~LRUCache() {}

bool LRUCache::lookUp(uint64_t addr, bool write) {
  auto set_idx = this->getSet(addr);
  auto tag = this->getTag(addr);

  for (uint32_t a = 0, n = 1 << assocBits_; a < n; ++a) {
    auto i = set_idx * n + a;
    if (tags_[i].valid && tags_[i].tag == tag) {
      tags_[i].dirty |= write ? 1 : 0;
      this->updatePos(set_idx, i);
      return true;
    }
  }
  return false;
}

std::pair<bool, uint64_t> LRUCache::insert(uint64_t addr, bool dirty) {
  auto set_idx = this->getSet(addr);
  auto tag = this->getTag(addr);

  std::pair<bool, uint64_t> ret(false, 0);
  int dest = -1;

  for (uint32_t a = 0, n = 1 << assocBits_, lru_pos = n - 1; a < n; ++a) {
    auto i = set_idx * n + a;
    if (tags_[i].valid) {
      if (tags_[i].pos == lru_pos) {
        if (tags_[i].dirty) {
          // return victim block address
          ret.first = true;
          ret.second = ((tags_[i].tag << setBits_) | set_idx) << blockBits_;
        }
        dest = i;
        break;
      }
    } else {
      // found free slot
      dest = i;
      assert(freeSpace_ != 0);
      --freeSpace_;
      break;
    }
  }

  // update cache
  assert(dest != -1);
  tags_[dest].tag = tag;
  tags_[dest].valid = 1;
  tags_[dest].dirty = dirty;
  this->updatePos(set_idx, dest);

  return ret;
}

void LRUCache::updatePos(uint32_t set_idx, uint32_t tag_idx) {
  uint32_t curr_pos = tags_[tag_idx].pos;
  for (uint32_t a = 0, n = 1 << assocBits_; a < n; ++a) {
    auto i = set_idx * n + a;
    if (tags_[i].pos < curr_pos) ++tags_[i].pos;
  }
  tags_[tag_idx].pos = 0;
}
