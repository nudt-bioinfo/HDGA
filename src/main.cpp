/**
 * @file    main.cpp
 * @author  Chirag Jain <cjain7@gatech.edu>
 */
#include <iostream>

#ifdef VTUNE_SUPPORT
#include <ittnotify.h>
#endif

#include "parseCmdArgs.hpp"
#include "graphLoad.hpp"
#include "align.hpp"
#include "utils.hpp"
#include "base_types.hpp"

int main(int argc, char **argv)
{
#ifdef VTUNE_SUPPORT
  __itt_pause();
#endif

  // parse command line arguments
  hdga::Parameters parameters;
  hdga::parseandSave(argc, argv, parameters);

  // buffer for results
  std::vector<hdga::BestScoreInfo> bestScoreVector;

  // execute alignment
  if (hdga::alignToDAG(parameters, hdga::MODE::LOCAL, bestScoreVector) == hdga_STATUS_OK)
    std::cout << "INFO, hdga::main, run finished" << std::endl;

  return 0;
}
