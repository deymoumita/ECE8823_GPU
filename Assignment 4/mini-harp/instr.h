#pragma once

#include "common.h"

namespace Harp {

enum Opcode {
  OP_NOP,
  OP_DI,
  OP_EI,
  OP_TLBADD,
  OP_TLBFLUSH,
  OP_NEG,
  OP_NOT,
  OP_AND,
  OP_OR,
  OP_XOR,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_MOD,
  OP_SHL,
  OP_SHR,
  OP_ANDI,
  OP_ORI,
  OP_XORI,
  OP_ADDI,
  OP_SUBI,
  OP_MULI,
  OP_DIVI,
  OP_MODI,
  OP_SHLI,
  OP_SHRI,
  OP_JALI,
  OP_JALR,
  OP_JMPI,
  OP_JMPR,
  OP_CLONE,
  OP_JALIS,
  OP_JALRS,
  OP_JMPRT,
  OP_LD,
  OP_ST,
  OP_LDI,
  OP_RTOP,
  OP_ANDP,
  OP_ORP,
  OP_XORP,
  OP_NOTP,
  OP_ISNEG,
  OP_ISZERO,
  OP_HALT,
  OP_TRAP,
  OP_JMPRU,
  OP_SKEP,
  OP_RETI,
  OP_TLBRM,
  OP_ITOF,
  OP_FTOI,
  OP_FADD,
  OP_FSUB,
  OP_FMUL,
  OP_FDIV,
  OP_FNEG,
  OP_WSPAWN,
  OP_SPLIT,
  OP_JOIN,
  OP_BAR
};

enum ArgClass {
  AC_NONE,
  AC_2REG,
  AC_2IMM,
  AC_3REG,
  AC_3PREG,
  AC_3IMM,
  AC_3REGSRC,
  AC_1IMM,
  AC_1REG,
  AC_3IMMSRC,
  AC_PREG_REG,
  AC_2PREG,
  AC_2REGSRC
};

enum InstType {
  ITYPE_NULL,
  ITYPE_INTBASIC,
  ITYPE_INTMUL,
  ITYPE_INTDIV,
  ITYPE_STACK,
  ITYPE_BR,
  ITYPE_CALL,
  ITYPE_RET,
  ITYPE_TRAP,
  ITYPE_FPBASIC,
  ITYPE_FPMUL,
  ITYPE_FPDIV
};

struct InstInfo {
  const char *opString;
  bool controlFlow;
  bool relAddress;
  bool allSrcArgs;
  bool privileged;
  ArgClass argClass;
  InstType iType;
};

class Instruction {
 public:
  Instruction(Opcode op);

  Opcode getOpcode() const { return op_; }

  bool hasPredReg() const { return predReg_ != 0xffffffff; }

  void setPredReg(unsigned reg) { predReg_ = reg; }

  unsigned getPredReg() const { return predReg_; }

  bool hasDestReg() const { return destReg_ != 0xffffffff; }

  void setDestReg(unsigned reg) { destReg_ = reg; }

  unsigned getDestReg() const { return destReg_; }

  bool hasDestPReg() const { return destPReg_ != 0xffffffff; }

  void setDestPReg(unsigned reg) { destPReg_ = reg; }

  unsigned getDestPReg() const { return destPReg_; }

  unsigned getSrcRegCount() const { return srcRegs_.size(); }

  void addSrcReg(unsigned reg) { srcRegs_.push_back(reg); }

  unsigned getSrcReg(unsigned i) const { return srcRegs_[i]; }

  unsigned getSrcPRegCount() const { return srcPRegs_.size(); }

  void addSrcPReg(unsigned reg) { srcPRegs_.push_back(reg); }

  unsigned getSrcPReg(unsigned i) const { return srcPRegs_[i]; }

  bool hasSrcImm() const { return hasImm_; }

  void setSrcImm(word_t value) {
    hasImm_ = true;
    immSrc_ = value;
  }

  word_t getSrcImm() const { return immSrc_; }

  const InstInfo &getInfo() const;

 private:
  Opcode op_;
  unsigned predReg_;
  unsigned destReg_;
  unsigned destPReg_;
  std::vector<unsigned> srcRegs_;
  std::vector<unsigned> srcPRegs_;
  bool hasImm_;
  word_t immSrc_;
  friend std::ostream &operator<<(std::ostream &, const Instruction &);
};

std::ostream &operator<<(std::ostream &, const Instruction &);
}  // namespace Harp
