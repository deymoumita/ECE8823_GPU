#include "utils.h"

using namespace Harp;

unsigned Harp::ceilLog2(unsigned x) {
  unsigned z = 0;
  bool nonZeroInnerValues(false);
  if (x == 0) {
    return 0;
  }
  while (x != 1) {
    z++;
    if (x & 1) {
      nonZeroInnerValues = true;
    }
    x >>= 1;
  }
  if (nonZeroInnerValues) {
    z++;
  }
  return z;
}

word_t Harp::bitMask(uint32_t bits) { return (1ull << bits) - 1; }

word_t Harp::signExt(word_t word, uint32_t bit, word_t mask) {
  if (word >> (bit - 1)) word |= ~mask;
  return word;
}
