#pragma once

#include "instr.h"

namespace Harp {

class Decoder {
 public:
  Decoder(unsigned wordsize, unsigned numGRegs, unsigned numPRegs);

  Instruction *decode(word_t code);

 protected:
  uint32_t wordsize_;
  uint32_t o_;
  uint32_t r_;
  uint32_t p_;
  uint32_t i1_;
  uint32_t i2_;
  uint32_t i3_;
  word_t oMask_;
  word_t rMask_;
  word_t pMask_;
  word_t i1Mask_;
  word_t i2Mask_;
  word_t i3Mask_;
};
}  // namespace Harp
