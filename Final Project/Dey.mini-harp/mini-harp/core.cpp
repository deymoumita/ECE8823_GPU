#include "core.h"

using namespace Harp;

Core::Core(unsigned num_warps, unsigned num_threads, unsigned num_regs,
           const char* output) {
  ram_ = new RamMemDevice(RAM_SIZE);
  console_ = new ConsoleDevice(output);
  mem_ = new MemoryUnit(PAGE_SIZE, ADDR_SIZE, true);
  mem_->attach(ram_, 0);
  mem_->attach(console_, CONSOLE_ADDR);
  icache_ = new Cache(ICACHE_SIZE, ICACHE_BLOCK, ICACHE_ASSOC, mem_);
  dcache_ = new Cache(DCACHE_SIZE, DCACHE_BLOCK, DCACHE_ASSOC, mem_);
  decoder_ = new Decoder(ADDR_SIZE * 8, num_regs, NUM_PREGS);
  for (unsigned i = 0; i < num_warps; ++i) {
    warps_.push_back(new Warp(this, i, num_threads, num_regs));
  }
  // enable Warp0
  warps_[0]->bootEnable();
  totalWarps = num_warps;
}

Core::~Core() {
  delete icache_;
  delete dcache_;
  delete decoder_;
  delete mem_;
  delete ram_;
  delete console_;
  for (auto warp : warps_) {
    delete warp;
  }
}

void Core::load(const char* program) { ram_->load(program); }

bool Core::running() const {
  for (auto warp : warps_) {
    if (warp->isActive()) return true;
  }
  return false;
}

void Core::step() {
  for (auto warp : warps_) {
    //if(warp->isActive())
      warp->step();
  }
}

void Core::printStats() {
  // dump performance counters
  std::ostream& os = *(std::ostream*)(console_->base());
  os << "Instruction Count: " << this->getInstructionCount() << std::endl;
}

uint64_t Core::getInstructionCount() const {
  uint64_t count = 0;
  for (auto warp : warps_) {
    count += warp->getInstructionCount();
  }
  return count;
}
