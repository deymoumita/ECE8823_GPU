#include "warp.h"
#include "core.h"
#include "utils.h"
#include <math.h>


using namespace Harp;

unsigned totalWarps[10000];
bool isFirstWarpAtBarrier[10000];

Warp::Warp(Core *core, unsigned id, unsigned numThreads, unsigned numRegs)
    : core_(core),
      id_(id),
      numThreads_(numThreads),
      numRegs_(numRegs),
      activeThreads_(0),
      pc_(0),
      spawned_(false),
      atBarrier_(false),
      barrierId_(-1),
      maxWarps_(0) {
  regFile_ = new RegFile(numThreads, numRegs);
  predRegs_.resize(numThreads);

  if(isFirstWarpAtBarrier[0] == false)
  {
    for(unsigned i=0; i<10000; ++i)
      isFirstWarpAtBarrier[i] = false;
  }

  prevActiveMask_.clear();
  for(unsigned i = 0; i < numThreads; ++i)
    prevActiveMask_.push_back(1);
  splitcount = 0;
  
}

Warp::~Warp() { delete regFile_; }

void Warp::bootEnable() {
  activeThreads_ = 1;
  spawned_ = true;
}

void Warp::step() {

  once = false;
  flag = false;

  // early exit if no thread is active
  if (0 == activeThreads_) return;

  // fetch next instruction
  auto code = this->fetch();

  // fetch next instruction
  auto instr = this->decode(code);

  // Update pc
  pc_ += ADDR_SIZE;

  // execute
  this->execute(instr);

  // Clean up.
  delete instr;
}

word_t Warp::fetch() {
  // fetch next instruction
  auto fetched = core_->icache()->read(pc_, false);
  return fetched.first;
}

Instruction *Warp::decode(word_t code) {
  // decode instruction
  auto instr = core_->decoder()->decode(code);
  D(3, "0x" << std::hex << pc_ << ": " << *instr);
  return instr;
}

void Warp::execute(Instruction *instr) {
  D(3, "Begin instruction execute.");

  auto nextActiveThreads = activeThreads_;
  auto nextPc = pc_;
  auto nextActiveMask_ = prevActiveMask_;


  for (unsigned t = 0; t < activeThreads_; ++t) {
    if (isActive())
      totalInstr_++;
    // skip false-predicated instructions
    if (instr->hasPredReg() && !predRegs_[t]) continue;
    if (!prevActiveMask_[t]) continue;

    std::cout<<std::endl<<"t="<<t;

     switch (instr->getOpcode()) {
      case OP_NOP:
        break;
      // TODO:
      case OP_ST: 
        core_->mem_->write(regFile_->get(t, instr->getSrcReg(1)) + instr->getSrcImm(), regFile_->get(t, instr->getSrcReg(0)), 1);
        break;
      case OP_LD: 
        regFile_->set(t, instr->getDestReg(), core_->mem_->read(regFile_->get(t, instr->getSrcReg(0)) + instr->getSrcImm(), 1));
        break;
      case OP_LDI: {
        regFile_->set(t, instr->getDestReg(), instr->getSrcImm());
        break;
        }
      case OP_ADDI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) + instr->getSrcImm());
        break;
      case OP_SUBI: {
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) - instr->getSrcImm());     
      break;
      }
      case OP_MULI:          
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) * instr->getSrcImm()); 
        break;
      case OP_SHLI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) << instr->getSrcImm());  
        break; 
      case OP_SHRI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) >> instr->getSrcImm());   
        break;
      case OP_ANDI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) & instr->getSrcImm());
        break;
      case OP_ORI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) | instr->getSrcImm());
        break;
      case OP_XORI: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) ^ instr->getSrcImm());
        break;  
      case OP_ADD: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) + regFile_->get(t, instr->getSrcReg(1)));
        break;
      case OP_SUB: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) - regFile_->get(t, instr->getSrcReg(1)));
        break;
      case OP_MUL: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) * regFile_->get(t, instr->getSrcReg(1)));
        break; 
      case OP_SHL: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) << regFile_->get(t, instr->getSrcReg(1)));
        break;
      case OP_SHR: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) >> regFile_->get(t, instr->getSrcReg(1)));
        break; 
      case OP_AND: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) & regFile_->get(t, instr->getSrcReg(1)));
        break;
      case OP_OR: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) | regFile_->get(t, instr->getSrcReg(1)));
        break;  
      case OP_XOR: 
        regFile_->set(t, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(0)) ^ regFile_->get(t, instr->getSrcReg(1)));
        break;
      case OP_NEG: 
        regFile_->set(t, instr->getDestReg(), (-regFile_->get(t, instr->getSrcReg(0)) + 1));
        break;
      case OP_NOT: 
        regFile_->set(t, instr->getDestReg(), (-regFile_->get(t, instr->getSrcReg(0))));
        break;
      case OP_CLONE: 
        regFile_->clone(t, regFile_->get(t, instr->getSrcReg(0)));
        break;
      case OP_BAR: {

        word_t bar_Id = regFile_->get(t, instr->getSrcReg(0));
        word_t max_warps = regFile_->get(t, instr->getSrcReg(1));

        // first thread to enter the barrier
        if(!atBarrier_)
        {
          this->threadAtBarrier_ = t;

        }
        if (t == this->threadAtBarrier_) 
        {
          this->atBarrier_ = true;

        }
        if (t == this->threadAtBarrier_ && this->atBarrier_ == true)
        {
          // first warp to enter barrier
          this->activeThreadsOld_ = this->activeThreads_;
          //this->nextActiveThreadsOld = nextActiveThreads;
          //this->nextActiveThreads_ = 0;
          nextActiveThreads = 0; 
          //this->activeThreads_ = 0;         
          this->barrierId_ = bar_Id;
          this->maxWarps_ = max_warps;          
          
          if(isFirstWarpAtBarrier[this->barrierId_] == false)
          {
            isFirstWarpAtBarrier[this->barrierId_] = true;
            totalWarps[this->barrierId_] = 0;
          }
          totalWarps[this->barrierId_]++;
          
          // last warp to enter
          if(totalWarps[this->barrierId_] == this->maxWarps_)
          {
            totalWarps[this->barrierId_] = 0;
            isFirstWarpAtBarrier[this->barrierId_] = false;
            for(unsigned m=0; m<core_->totalWarps; ++m)
            {
              if((core_->warps_[m]->barrierId_ == this->barrierId_) && (core_->warps_[m]->atBarrier_ == true))
              {
                core_->warps_[m]->activeThreads_ = core_->warps_[m]->activeThreadsOld_;
                //core_->warps_[m]->nextActiveThreads = core_->warps_[m]->nextActiveThreadsOld;
                core_->warps_[m]->barrierId_ = -1;
                core_->warps_[m]->threadAtBarrier_ = 10000;
                core_->warps_[m]->atBarrier_ = false;
                core_->warps_[m]->maxWarps_ = 0;
              }
            }
          }
        }
        break;
      }
      case OP_WSPAWN:
        for (auto warp : core_->warps()) {
          if (!warp->spawned_) {
            warp->pc_ = regFile_->get(t, instr->getSrcReg(0));
            warp->regFile_->set(0, instr->getDestReg(), regFile_->get(t, instr->getSrcReg(1)));
            warp->activeThreads_ = 1;
            warp->spawned_ = true;
            break;
          }
        }
        break;
      case OP_JMPI:
        if (!once) 
        {
          once = true;
          nextPc = pc_ + instr->getSrcImm();
        }
        break;
      case OP_JALI:
        regFile_->set(t, instr->getDestReg(), pc_);
        if (!once) 
        {
          once = true;
          nextPc = pc_ + instr->getSrcImm();
        }
        break;
      case OP_JALR:
        regFile_->set(t, instr->getDestReg(), pc_);
        if (!once) 
        {
          once = true;
          nextPc = regFile_->get(t, instr->getSrcReg(0));
        }
        break;
      case OP_JMPR:
        if (!once) 
        {
          once = true;
          nextPc = regFile_->get(t, instr->getSrcReg(0));
        }
        break;
      case OP_JALIS:
        nextActiveThreads = regFile_->get(t, instr->getSrcReg(0));
        regFile_->set(t, instr->getDestReg(), pc_);
        if (!once) 
        {
          once = true;
          nextPc = pc_ + instr->getSrcImm();
        }
        break;
      case OP_JALRS:
        nextActiveThreads = regFile_->get(t, instr->getSrcReg(0));
        regFile_->set(t, instr->getDestReg(), pc_);
        if (!once) 
        {
          once = true;
          nextPc = regFile_->get(t, instr->getSrcReg(1));
        }
        break;
      case OP_JMPRT:
        nextActiveThreads = 1;
        if (!once) 
        {
          once = true;
          nextPc = regFile_->get(t, instr->getSrcReg(0));
        }
        break;
      case OP_RTOP: {
        predRegs_[t] = regFile_->get(t, instr->getSrcReg(0));
        //std::cout<<std::endl<< regFile_->get(t, instr->getSrcReg(0)) << std::endl;
        break;
      }
      case OP_ISZERO: {
        predRegs_[t] = !regFile_->get(t, instr->getSrcReg(0));
        break;
      }
      case OP_ISNEG: {        
        predRegs_[t] = regFile_->get(t, instr->getSrcReg(0)) & 0x80000000;
        break;
      }
      case OP_NOTP: {
        predRegs_[t] = !predRegs_[t];
        break;
      }
      case OP_HALT: 
        nextActiveThreads = 0;
        break;
      case OP_SPLIT: {
        if(!once)
        {
          once = true;
          std::vector<bool>::iterator it;
          unsigned it_i = 0;
          std::vector<bool> splitActiveMask_;
          splitActiveMask_.clear();
          // current predicate reg = true pred split mask; 
          for(it = predRegs_.begin(); it != predRegs_.end(); it++, it_i++)    
          {
            splitActiveMask_.push_back(predRegs_[it_i]);
          }

          // print for debug
          std::cout<<std::endl<<"Current Mask: ";
          for(it = prevActiveMask_.begin(); it != prevActiveMask_.end(); it++)  
          {
            std::cout<<*it;
          }       
          std::cout<<std::endl;   
          std::cout<<std::endl<<"Mask after split: ";        
          it_i = 0;
          for(it = splitActiveMask_.begin(); it != splitActiveMask_.end(); it++, ++it_i)
          {
            std::cout<<*it;
            // handles last thread entering split in divergent loop
            // assigns its active mask to 1111
            if(splitActiveMask_[it_i] == 0)
            {
              flag = flag | false;
            }      
            else
            {
              flag = flag | true;
            }              
          }       
          std::cout<<std::endl; 

          // if it's the last thread of divergent loop
          if (flag == false)
          {
            cd_npc_.clear();
            cd_mask_.clear();
            cd_npc_.push_back(nextPc); // dummy
            cd_npc_.push_back(nextPc + ADDR_SIZE); // false predicate
            std::vector<bool> tempmask;
            for(unsigned tempcount=0; tempcount<numThreads_; tempcount++)
              tempmask.push_back(1);
            cd_mask_.push_back(tempmask); // dummy
            cd_mask_.push_back(tempmask);

            std::cout<<"EXIT loop split";
            continue;
          }

          // Calculate invert of split mask
          std::vector<bool> invSplitActiveMask;
          invSplitActiveMask.clear();
          it_i = 0;
          for(it = splitActiveMask_.begin(); it != splitActiveMask_.end(); it++, ++it_i)
          {
            // ANDs parent mask and !current mask
            invSplitActiveMask.push_back(prevActiveMask_[it_i] & !splitActiveMask_[it_i]); 
            //std::cout<<std::endl<<prevActiveMask_[it_i]<<!splitActiveMask_[it_i]<<(prevActiveMask_[it_i] & !splitActiveMask_[it_i]);
          }
          // print for debug purpose
          std::cout<<std::endl<<"Mask after split invert: ";
          for(it = invSplitActiveMask.begin(); it != invSplitActiveMask.end(); it++, ++it_i)
          {
            std::cout<<*it;
          }       
          std::cout<<std::endl;

          // if it's the first split instruction, simply push into stack
          if (cd_npc_.empty())
          {
            cd_npc_.push_back(nextPc);
            cd_npc_.push_back(nextPc + ADDR_SIZE);
            cd_mask_.push_back(splitActiveMask_);
            cd_mask_.push_back(invSplitActiveMask);
          }
          // if nested splits, insert on TOS
          else
          { 
            cd_mask_.insert(cd_mask_.begin(), invSplitActiveMask);         
            cd_mask_.insert(cd_mask_.begin(), splitActiveMask_);           
            cd_npc_.insert(cd_npc_.begin(), nextPc + ADDR_SIZE);
            cd_npc_.insert(cd_npc_.begin(), nextPc);
          } 
          
          // activate next threads based on split predicate 
          nextActiveMask_ = splitActiveMask_; 
        }
        break;
      }
      case OP_JOIN: {
        if(!once)
        {
          once = true;
          splitcount++;
          std::vector<bool>::iterator it;
          int it_i = 0;

          // pop TOS
          cd_npc_.erase(cd_npc_.begin());
          cd_mask_.erase(cd_mask_.begin());

          // both true and false haven't been executed, go to the false one from TOS
          if (splitcount < 2)
          {
            nextPc = cd_npc_[0];
            nextActiveMask_ = cd_mask_[0];
          }
          // this split-join is over, so execute the next instruction with the TOS active mask
          else
          {
            it_i = 0;
            if (!cd_mask_.empty())
              nextActiveMask_ = cd_mask_[0];
            // if TOS is empty, all threads are activated
            else
            {
              for(it = nextActiveMask_.begin(); it != nextActiveMask_.end(); it++, it_i++)
                nextActiveMask_[it_i] = 1;
            }
            
            splitcount = 0;
          }
            
          std::cout<<std::endl<<"New Mask: ";
          for(it = nextActiveMask_.begin(); it != nextActiveMask_.end(); it++)  
          {
            std::cout<< *it;  // prints vector value
          }  
        }
        break;
      }    
      // New instruction that reads threadID 
      case OP_TID:
         regFile_->set(t, instr->getSrcReg(0), t);
         //std::cout<<"\n"<<regFile_->get(t, instr->getSrcReg(0))<<std::endl;
         break;
      default:
        std::cout << "ERROR: Unsupported instruction: " << *instr << "\n";
        exit(EXIT_FAILURE);
    }
  }

  if (nextActiveThreads > numThreads_) {
    std::cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
              << numThreads_ << " available.\n";
    abort();
  }

  if(instr->getOpcode() != OP_BAR )
  {
    activeThreads_ = nextActiveThreads;
    pc_ = nextPc;
    prevActiveMask_ = nextActiveMask_;
  }

  D(3, "End instruction execute.");
}

uint64_t Warp::getInstructionCount() const {
  // TODO:

  return totalInstr_;
}
