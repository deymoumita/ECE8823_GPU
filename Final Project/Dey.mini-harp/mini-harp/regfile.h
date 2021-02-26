#include "common.h"

namespace Harp {

class RegFile {
 public:
  RegFile(unsigned numThreads, unsigned numRegs) {
    // TODO
    _numRegs = numRegs;
    _regFile = new word_t *[numThreads];
    for(unsigned t_count=0; t_count<numThreads; ++t_count)
      _regFile[t_count] = new word_t[numRegs];
  }
  ~RegFile() {delete *_regFile;}

  word_t get(unsigned threadIdx, unsigned regNum) const {
    // TODO
    return _regFile[threadIdx][regNum];
  }

  void set(unsigned threadIdx, unsigned regNum, word_t value) {
    // TODO
    _regFile[threadIdx][regNum] = value;
  }

  void clone(unsigned lane_src, unsigned lane_dest)
  {
    //printf("%d -> %d\n", lane_src, lane_dest);
    delete _regFile[lane_dest];
    _regFile[lane_dest] = new word_t[_numRegs];
    for(unsigned reg_count=0; reg_count<_numRegs; ++reg_count)
    {
      _regFile[lane_dest][reg_count] = _regFile[lane_src][reg_count];
      //printf("%d ", reg_count);
    }

  }

  private:
    word_t **_regFile;
    unsigned _numRegs;
};

}  // namespace Harp
