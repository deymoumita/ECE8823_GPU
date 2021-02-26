#pragma once

#include "instr.h"
#include "regfile.h"

namespace Harp {

class Core;

class Warp {
 public:
  Warp(Core *core, unsigned id, unsigned numThreads, unsigned numRegs);
  ~Warp();

  void bootEnable();

  bool isActive() const { return (activeThreads_ != 0); }

  void step();

  uint64_t getInstructionCount() const;

 private:
  word_t fetch();

  Instruction *decode(word_t code);

  void execute(Instruction *instr);

  Core *core_;

  RegFile *regFile_;

  std::vector<bool> predRegs_;

  unsigned id_;
  unsigned numThreads_;
  unsigned numRegs_;
  unsigned activeThreads_;
  unsigned pc_;
  bool spawned_;
  uint64_t totalInstr_;

  bool atBarrier_;
  word_t barrierId_;
  unsigned activeThreadsOld_;
  unsigned threadAtBarrier_;
  word_t maxWarps_;
  //unsigned nextActiveThreads;
  //unsigned nextActiveThreadsOld;


};
}  // namespace Harp
