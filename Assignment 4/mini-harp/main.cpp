#include <getopt.h>
#include <stdio.h>
#include "core.h"

using namespace Harp;

void showUsage() {
  printf("Usage: [options] <program file>\n");
  printf("Options: -r <number of registers>\n");
  printf("         -t <threads per warp>\n");
  printf("         -w <number of warps>\n");
  printf("         -o <output file>\n");
}

int main(int argc, char *argv[]) {
  int num_threads = NUM_TREADS;
  int num_warps = NUM_WARPS;
  int num_regs = NUM_GREGS;
  char *program = nullptr;
  char *output = nullptr;

  // parse command line arguments
  int c;
  while ((c = getopt(argc, argv, "r:t:w:o:")) != -1) {
    switch (c) {
      case 'r':
        num_regs = atoi(optarg);
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'w':
        num_warps = atoi(optarg);
        break;
      case 'o':
        output = optarg;
        break;
      case '?':
      case 'h':
      default:
        showUsage();
        exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    program = argv[optind];
  } else {
    showUsage();
    exit(EXIT_FAILURE);
  }

  {
    Core core(num_warps, num_threads, num_regs, output);

    // load program
    core.load(program);

    // execute program
    while (core.running()) {
      core.step();
    }

    // print stats
    core.printStats();
  }

  D(3, "Done.");

  return 0;
}
