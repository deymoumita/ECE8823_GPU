#include "decode.h"
#include "utils.h"

using namespace Harp;

Decoder::Decoder(unsigned wordsize, unsigned numGRegs, unsigned numPRegs) {
  wordsize_ = wordsize;
  o_ = 6;
  r_ = ceilLog2(numGRegs);
  p_ = ceilLog2(numPRegs);
  i1_ = wordsize_ - 1 - p_ - o_;
  i2_ = i1_ - r_;
  i3_ = i2_ - r_;
  oMask_ = bitMask(o_);
  rMask_ = bitMask(r_);
  pMask_ = bitMask(p_);
  i1Mask_ = bitMask(i1_);
  i2Mask_ = bitMask(i2_);
  i3Mask_ = bitMask(i3_);
}

Instruction *Decoder::decode(word_t code) {
  Opcode op = (Opcode)((code >> i1_) & oMask_);
  Instruction *instr = new Instruction(op);

  bool predicated = code >> (wordsize_ - 1);
  if (predicated) {
    instr->setPredReg((code >> (wordsize_ - p_ - 1)) & pMask_);
  }

  const InstInfo &info = instr->getInfo();
  switch (info.argClass) {
    case AC_NONE:
      break;
    case AC_1IMM:
      instr->setSrcImm(signExt(code & i1Mask_, i1_, i1Mask_));
      break;
    case AC_2IMM:
      instr->setDestReg((code >> i2_) & rMask_);
      instr->setSrcImm(signExt(code & i2Mask_, i2_, i2Mask_));
      break;
    case AC_3IMM:
      instr->setDestReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      instr->setSrcImm(signExt(code & i3Mask_, i3_, i3Mask_));
      break;
    case AC_3IMMSRC:
      instr->addSrcReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      instr->setSrcImm(signExt(code & i3Mask_, i3_, i3Mask_));
      break;
    case AC_1REG:
      instr->addSrcReg((code >> i2_) & rMask_);
      break;
    case AC_2REG:
      instr->setDestReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      break;
    case AC_3REG:
      instr->setDestReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      instr->addSrcReg((code >> (i3_ - r_)) & rMask_);
      break;
    case AC_3REGSRC:
      instr->addSrcReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      instr->addSrcReg((code >> (i3_ - r_)) & rMask_);
      break;
    case AC_PREG_REG:
      instr->setDestPReg((code >> i2_) & pMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      break;
    case AC_2PREG:
      instr->setDestPReg((code >> i2_) & pMask_);
      instr->addSrcPReg((code >> i3_) & pMask_);
      break;
    case AC_3PREG:
      instr->setDestPReg((code >> i2_) & pMask_);
      instr->addSrcPReg((code >> i3_) & pMask_);
      instr->addSrcPReg((code >> (i3_ - r_)) & pMask_);
      break;
    case AC_2REGSRC:
      instr->addSrcReg((code >> i2_) & rMask_);
      instr->addSrcReg((code >> i3_) & rMask_);
      break;
    default:
      std::cout << "Unrecognized argument class in word decoder.\n";
      //std::cout<<info.argClass;
      exit(EXIT_FAILURE);
  }

  D(2, "Decoded 0x" << std::hex << code << " into: " << *instr << '\n');

  return instr;
}
