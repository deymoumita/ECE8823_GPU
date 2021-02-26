CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -DUSE_DEBUG=3
LDFLAGS += -lstdc++

PROJECT = miniharp.out

SRCS = utils.cpp instr.cpp decode.cpp lrucache.cpp mem.cpp warp.cpp core.cpp main.cpp

OBJS = $(SRCS:.cpp=.o)

all : clean $(PROJECT)


$(PROJECT) : $(OBJS) 
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.depend : $(SRCS) 
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean: 
	rm -f *.out *.o *~ *\#

run : $(PROJECT)
	. /$(PROJECT) simple.bin

ifneq ($(MAKECMDGOALS), clean) 
	-include.depend 
endif
