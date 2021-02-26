#pragma once

#include "common.h"
#include "lrucache.h"

namespace Harp {

class MemDevice {
 public:
  MemDevice() {}
  virtual ~MemDevice() {}

  virtual uint32_t size() const = 0;
  virtual byte_t* base() = 0;

  virtual word_t read(addr_t addr) = 0;
  virtual void write(addr_t addr, word_t data) = 0;
};

class RamMemDevice : public MemDevice {
 public:
  RamMemDevice(uint32_t size);

  void load(const char* filename);

  uint32_t size() const override { return contents_.size(); };
  byte_t* base() override { return &contents_[0]; }

  word_t read(addr_t addr) override;
  void write(addr_t addr, word_t data) override;

 protected:
  std::vector<byte_t> contents_;
};

class ConsoleDevice : public MemDevice {
 public:
  ConsoleDevice(const char* output);
  ~ConsoleDevice();

  uint32_t size() const override { return ADDR_SIZE; }
  byte_t* base() override { return (byte_t*)os_; }

  word_t read(addr_t) override { return 0; }
  void write(addr_t, word_t data) override;

 private:
  std::ostream* os_;
};

class MemoryUnit {
 public:
  MemoryUnit(uint32_t pageSize, uint32_t addrBytes, bool disableVM);

  void attach(MemDevice* md, addr_t base);

  struct PageFault {
    PageFault(addr_t vAddr, bool notFound)
        : faultAddr(vAddr), notFound(notFound) {}
    addr_t faultAddr;
    bool notFound;
  };

  word_t read(addr_t vAddr, bool supervisorMode);

  void write(addr_t vAddr, word_t data, bool supervisorMode);

  void tlbAdd(addr_t vAddr, addr_t phys, uint32_t flags);
  void tlbRm(addr_t vAddr);
  void tlbFlush();

 private:
  struct mapping_t {
    addr_t base;
    addr_t end;
    MemDevice* dev;
  };

  struct tlb_entry_t {
    uint32_t frameIdx;
    uint32_t flags;
  };

  const mapping_t* getMapping(addr_t addr);

  const tlb_entry_t* tlbLookup(addr_t vAddr, uint32_t flagMask);

  uint32_t pageSize_;
  uint32_t addrBytes_;
  bool disableVM_;
  std::list<mapping_t> mappings_;
  std::map<addr_t, tlb_entry_t> tlb_;
};

class Cache {
 public:
  Cache(uint64_t capacity, uint32_t blockSize, uint32_t associativity,
        MemoryUnit* mem);

  std::pair<word_t, bool> read(addr_t vAddr, bool supervisorMode);
  bool write(addr_t vAddr, word_t data, bool supervisorMode);

 private:
  LRUCache lrucache_;
  MemoryUnit* mem_;
};

}  // namespace Harp
