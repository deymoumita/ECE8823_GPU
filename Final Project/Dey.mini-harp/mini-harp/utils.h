#include "common.h"

namespace Harp {

unsigned ceilLog2(unsigned x);

word_t bitMask(uint32_t bits);

word_t signExt(word_t word, uint32_t bit, word_t mask);

}  // namespace Harp
