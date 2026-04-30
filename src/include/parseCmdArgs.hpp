/**
 * @file    parseCmdArgs.hpp
 * @brief   command line parsing
 * @author  Chirag Jain <cjain7@gatech.edu>
 */

#ifndef PARSE_CMD_HPP
#define PARSE_CMD_HPP

#include "base_types.hpp"
#include "utils.hpp"
#include "clipp.h"

namespace hdga
{
  /**
   * @brief                   parse the cmd line options
   * @param[in]   argc
   * @param[in]   argv
   * @param[out]  param       parameters are saved here
   **/
  void parseandSave(int argc, char **argv, hdga::Parameters &param)
  {
    // set the default scoring scheme if not modified later
    param.match = param.mismatch = param.ins = param.del = 1;
    param.threads = 1;
    param.banded = true;
    param.hybridCpuGpu = false;
    param.cpuSpeed = 1.0;
    param.gpuSpeed = 1.0;

    // define all arguments
    auto cli =
        (clipp::required("-m") &
             (clipp::required("vg").set(param.mode) | clipp::required("txt").set(param.mode)).doc("reference graph format"),
         clipp::required("-r") & clipp::value("ref", param.rfile).doc("reference graph file"),
         clipp::required("-q") & clipp::value("query", param.qfile).doc("query file (fasta/fastq)[.gz]"),
         clipp::required("-o") & clipp::value("output", param.ofile).doc("output file"),
         clipp::required("-t") & clipp::value("threads", param.threads).doc("thread count for parallel execution"),
         clipp::option("--disable-banded").call([&param]()
                                                { param.banded = false; })
             .doc("disable banded strategy (enabled by default)"),
         clipp::option("--hybrid-cpu-gpu").call([&param]()
                                                { param.hybridCpuGpu = true; })
             .doc("enable runtime hybrid CPU/GPU scheduling"),
         clipp::option("--cpu-speed") & clipp::value("S1", param.cpuSpeed).doc("relative CPU speed weight for hybrid scheduling (default 1.0)"),
         clipp::option("--gpu-speed") & clipp::value("S2", param.gpuSpeed).doc("relative GPU speed weight for hybrid scheduling (default 1.0)"),
         clipp::option("-match") & clipp::value("N1", param.match).doc("match score (default 1)"),
         clipp::option("-mismatch") & clipp::value("N2", param.mismatch).doc("mismatch penalty (default 1)"),
         clipp::option("-ins") & clipp::value("N3", param.ins).doc("insertion penalty (default 1)"),
         clipp::option("-del") & clipp::value("N4", param.del).doc("deletion penalty (default 1)"));

    if (!clipp::parse(argc, argv, cli))
    {
      // print help page
      clipp::operator<<(std::cout, clipp::make_man_page(cli, argv[0])) << std::endl;
      exit(1);
    }

    omp_set_num_threads(param.threads);

    // print execution environment based on which MACROs are set
    hdga::showExecutionEnv();
  }
}

#endif
