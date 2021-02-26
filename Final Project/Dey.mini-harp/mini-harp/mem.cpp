#include "mem.h"
#include "utils.h"

using namespace Harp;

RamMemDevice::RamMemDevice(uint32_t size) : contents_(size) {}

void RamMemDevice::load(const char* filename) {
  std::ifstream input(filename);
  if (!input) {
    std::cout << "Couldn't access file '" << filename << "'." << std::endl;
    exit(EXIT_FAILURE);
  }

  // copy bytes
  size_t offset = 0;
  do {
    if (offset >= contents_.size()) {
      std::cout << "RAM out of space." << std::endl;
      exit(EXIT_FAILURE);
    }
    contents_[offset++] = input.get();
  } while (input);

  // add padding
  while (contents_.size() % ADDR_SIZE) {
    if (offset >= contents_.size()) {
      std::cout << "RAM out of space." << std::endl;
      exit(EXIT_FAILURE);
    }
    contents_[offset++] = 0x00;
  }
}

word_t RamMemDevice::read(addr_t addr) {
  D(2, "RAM read, addr=0x" << std::hex << addr);
  auto wordSize(ADDR_SIZE - addr % ADDR_SIZE);
  if (addr + wordSize > contents_.size()) {
    throw std::out_of_range("offset out of range");
  }
  word_t w(0);
  addr += wordSize;
  for (uint32_t i = 0; i < wordSize; i++) {
    w <<= 8;
    w |= contents_[addr - i - 1];
  }
  return w;
}

void RamMemDevice::write(addr_t addr, word_t data) {
  D(2, "RAM write, addr=0x" << std::hex << addr);
  auto wordSize(ADDR_SIZE - addr % ADDR_SIZE);
  if (addr + wordSize > contents_.size()) {
    throw std::out_of_range("offset out of range");
  }
  while (wordSize--) {
    contents_[addr++] = data & 0xff;
    data >>= 8;
  }
}

///////////////////////////////////////////////////////////////////////////////

ConsoleDevice::ConsoleDevice(const char* output) {
  if (output) {
    os_ = new std::ofstream(output);
  } else {
    os_ = &std::cout;
  }
}

ConsoleDevice::~ConsoleDevice() {
  std::flush(*os_);
  if (os_ != &std::cout) {
    delete os_;
  }
}

void ConsoleDevice::write(addr_t, word_t data) { *os_ << char(data); }

///////////////////////////////////////////////////////////////////////////////

MemoryUnit::MemoryUnit(uint32_t pageSize, uint32_t addrBytes, bool disableVM)
    : pageSize_(pageSize), addrBytes_(addrBytes), disableVM_(disableVM) {
  if (!disableVM) {
    tlb_[0] = tlb_entry_t{0, 0x77};
  }
}

void MemoryUnit::attach(MemDevice* md, addr_t base) {
  // check for range conflict
  auto end = base + md->size();
  if ((base % addrBytes_) != 0 || (end % addrBytes_) != 0) {
    std::cout << "Unaligned memory device address range." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (auto& mapping : mappings_) {
    if (base < mapping.end && end > mapping.base) {
      std::cout << "Address range conflicts with existing memory device."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // add new mapping
  mappings_.push_back(mapping_t{base, end, md});
}

const MemoryUnit::mapping_t* MemoryUnit::getMapping(addr_t addr) {
  auto end = addr + addrBytes_;
  for (auto& mapping : mappings_) {
    if (addr < mapping.end && end > mapping.base) {
      return &mapping;
    }
  }
  D(2, "No memory device mapped at address 0x" << std::hex << addr << ".");
  throw std::out_of_range("Bad address");
}

word_t MemoryUnit::read(addr_t vAddr, bool supervisorMode) {
  addr_t pAddr;
  if (disableVM_) {
    pAddr = vAddr;
  } else {
    uint32_t flagMask = supervisorMode ? 4 : 1;
    auto tlbEntry = this->tlbLookup(vAddr, flagMask);
    pAddr = tlbEntry->frameIdx * pageSize_ + vAddr % pageSize_;
  }
  auto mapping = getMapping(pAddr);
  return mapping->dev->read(pAddr - mapping->base);
}

void MemoryUnit::write(addr_t vAddr, word_t data, bool supervisorMode) {
  addr_t pAddr;
  if (disableVM_) {
    pAddr = vAddr;
  } else {
    uint32_t flagMask = supervisorMode ? 8 : 2;
    auto tlbEntry = this->tlbLookup(vAddr, flagMask);
    pAddr = tlbEntry->frameIdx * pageSize_ + vAddr % pageSize_;
  }
  auto mapping = getMapping(pAddr);
  mapping->dev->write(pAddr - mapping->base, data);
}

void MemoryUnit::tlbAdd(addr_t vAddr, addr_t phys, uint32_t flags) {
  D(3, "tlbAdd(0x" << std::hex << vAddr << ", 0x" << phys << ", 0x" << flags
                   << ')');
  uint32_t pageIdx(vAddr / pageSize_);
  uint32_t frameIdx(phys / pageSize_);
  tlb_[pageIdx] = tlb_entry_t{frameIdx, flags};
}

void MemoryUnit::tlbRm(addr_t vAddr) {
  D(3, "tlbRm(0x" << std::hex << vAddr << ')');
  auto pageIdx = vAddr / pageSize_;
  auto it = tlb_.find(pageIdx);
  if (it != tlb_.end()) {
    tlb_.erase(it);
  }
}

void MemoryUnit::tlbFlush() { tlb_.clear(); }

const MemoryUnit::tlb_entry_t* MemoryUnit::tlbLookup(addr_t vAddr,
                                                     uint32_t flagMask) {
  auto pageIdx = vAddr / pageSize_;
  auto it = tlb_.find(pageIdx);
  if (it != tlb_.end()) {
    const tlb_entry_t& t = it->second;
    if (t.flags & flagMask) {
      return &t;
    } else {
      D(2, "Page fault at 0x" << std::hex << vAddr << "(bad flags)");
      throw PageFault(vAddr, false);
    }
  } else {
    D(2, "Page fault at 0x" << std::hex << vAddr << "(not in TLB)");
    throw PageFault(vAddr, true);
  }
}

///////////////////////////////////////////////////////////////////////////////

Cache::Cache(uint64_t capacity, uint32_t blockSize, uint32_t associativity,
             MemoryUnit* mem)
    : lrucache_(capacity, blockSize, associativity), mem_(mem) {}

std::pair<word_t, bool> Cache::read(addr_t vAddr, bool supervisorMode) {
  bool hit = lrucache_.lookUp(vAddr, false);
  auto value = mem_->read(vAddr, supervisorMode);
  return std::make_pair(value, hit);
}

bool Cache::write(addr_t vAddr, word_t data, bool supervisorMode) {
  bool hit = lrucache_.lookUp(vAddr, true);
  if (!hit) {
    lrucache_.insert(vAddr, true);
  }
  mem_->write(vAddr, data, supervisorMode);
  return hit;
}
