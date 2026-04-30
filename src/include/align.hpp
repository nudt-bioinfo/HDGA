/**
 * @file    align.hpp
 * @brief   routines to perform alignment
 * @author  Chirag Jain <cjain7@gatech.edu>, Qingzhe Wang <1197460504@qq.com>
 */

#ifndef GRAPH_ALIGN_HPP
#define GRAPH_ALIGN_HPP

#include <immintrin.h>
#include <x86intrin.h>
#include <zlib.h>

#include "graphLoad.hpp"
#include "csr_char.hpp"
#include "graph_iter.hpp"
#include "base_types.hpp"
#include "utils.hpp"

#include "align_vectorized_difference.hpp"
#include "align_vectorized_banded.hpp"
#include "align_nonbanded.hpp"

#if defined(USE_GPU_PHASE1) || defined(USE_GPU_PHASE2)
#include "align_vectorized_gpu.hpp"
#endif
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

// External includes
#include "kseq/kseq.h"

KSEQ_INIT(gzFile, gzread)

#define hdga_max(a, b) (((a) > (b)) ? (a) : (b))

namespace hdga
{
    /**
     * @brief                               local alignment routine
     * @param[in]   readSet
     * @param[in]   graph                   node-labeled directed graph
     * @param[in]   parameters              input parameters
     * @param[out]  outputBestScoreVector
     */
    void alignToDAGLocalWithBackend(const std::vector<std::string> &readSet,
                                    const CSR_char_container &graph,
                                    const Parameters &parameters,
                                    std::vector<BestScoreInfo> &outputBestScoreVector,
                                    const bool useGpuPhase1,
                                    const bool useGpuPhase2,
                                    const std::vector<int32_t> *globalQryIds)
    {
        // create buffer to save best score info for each read and its rev. complement
        std::vector<BestScoreInfo> bestScoreVector_P1(2 * readSet.size());

        // to save input query sequences for phase 1
        std::vector<std::string> readSet_P1;

        assert(readSet.size() > 0);
        assert(outputBestScoreVector.empty());

        /* [Phase1] forward computing: get best score values and ending locations */
        {
            PhaseTimer t1("Phase1: forward computing", 0);

            size_t maxReadLength = 0;

            for (size_t readno = 0; readno < readSet.size(); readno++)
            {
                std::string read_reverse(readSet[readno]);
                hdga::seqUtils::reverseComplement(readSet[readno], read_reverse);

                readSet_P1.push_back(readSet[readno]);
                readSet_P1.push_back(read_reverse);

                if (readSet[readno].length() > maxReadLength)
                    maxReadLength = readSet[readno].length();
            }

            assert(bestScoreVector_P1.size() == 2 * readSet.size());
            assert(readSet_P1.size() == 2 * readSet.size());

#if defined(USE_GPU_PHASE1)
            if (useGpuPhase1)
            {
                t1.set_method("GPU vectorized");
                Phase1_Vectorized_GPU<int8_t> aligner_Phase1(readSet_P1, graph, parameters);
                aligner_Phase1.alignToDAGLocal_Phase1_vectorized_wrapper(bestScoreVector_P1);
            }
            else
            {
                // align read to ref.
                // there would be few padded characters at the end of qry seq
                // take that into account when computing max. read length
                auto blockHeight = Phase1_Vectorized<SimdInst<int8_t>>::blockHeight;
                maxReadLength += blockHeight - 1 - (maxReadLength - 1) % blockHeight;

                // decide precision by looking at maximum score possible
                if (maxReadLength * parameters.match <= INT8_MAX)
                {
                    t1.set_method("vectorized <8bit per cell>");
                    Phase1_Vectorized<SimdInst<int8_t>> aligner_Phase1(readSet_P1, graph, parameters);
                    aligner_Phase1.alignToDAGLocal_Phase1_vectorized_wrapper(bestScoreVector_P1);
                }
                else if (maxReadLength * parameters.match <= INT16_MAX)
                {
                    t1.set_method("vectorized <16bit per cell>");
                    Phase1_Vectorized<SimdInst<int16_t>> aligner_Phase1(readSet_P1, graph, parameters);
                    aligner_Phase1.alignToDAGLocal_Phase1_vectorized_wrapper(bestScoreVector_P1);
                }
                // when INT32 is needed, use difference recurrence relations
                else
                {
                    t1.set_method("difference vectorized <8bit per cell>");
                    Phase1_Vectorized_Difference<SimdInst<int8_t>> aligner_Phase1(readSet_P1, graph, parameters);
                    aligner_Phase1.alignToDAGLocal_Phase1_vectorized_difference_wrapper(bestScoreVector_P1);
                }
            }
#else
            (void)useGpuPhase1;
            // align read to ref.
            // there would be few padded characters at the end of qry seq
            // take that into account when computing max. read length
            auto blockHeight = Phase1_Vectorized<SimdInst<int8_t>>::blockHeight;
            maxReadLength += blockHeight - 1 - (maxReadLength - 1) % blockHeight;

            // decide precision by looking at maximum score possible
            if (maxReadLength * parameters.match <= INT8_MAX)
            {
                t1.set_method("vectorized <8bit per cell>");
                Phase1_Vectorized<SimdInst<int8_t>> aligner_Phase1(readSet_P1, graph, parameters);
                aligner_Phase1.alignToDAGLocal_Phase1_vectorized_wrapper(bestScoreVector_P1);
            }
            else if (maxReadLength * parameters.match <= INT16_MAX)
            {
                t1.set_method("vectorized <16bit per cell>");
                Phase1_Vectorized<SimdInst<int16_t>> aligner_Phase1(readSet_P1, graph, parameters);
                aligner_Phase1.alignToDAGLocal_Phase1_vectorized_wrapper(bestScoreVector_P1);
            }
            // when INT32 is needed, use difference recurrence relations
            else
            {
                t1.set_method("difference vectorized <8bit per cell>");
                Phase1_Vectorized_Difference<SimdInst<int8_t>> aligner_Phase1(readSet_P1, graph, parameters);
                aligner_Phase1.alignToDAGLocal_Phase1_vectorized_difference_wrapper(bestScoreVector_P1);
            }
#endif
        }

#ifdef DEBUG
        {
            for (size_t readno = 0; readno < readSet_P1.size(); readno++)
            {
                std::cout << "INFO, hdga::alignToDAGLocal, read # " << readno + 1 << ", score = " << bestScoreVector_P1[readno].score
                          << ", refColumnEnd = " << bestScoreVector_P1[readno].refColumnEnd
                          << ", qryRowEnd = " << bestScoreVector_P1[readno].qryRowEnd << "\n";
            }
        }
#endif

        /* [Phase2] backward computing & traceback: get begin locations and compute base-to-base alignment in cigar format */
        {
            PhaseTimer t2("Phase2: backward computing & traceback", 0);
            std::vector<std::string> readSet_P2;

            size_t maxReadLength = 0;

            for (size_t readno = 0; readno < readSet.size(); readno++)
            {
                if (bestScoreVector_P1[2 * readno].score > bestScoreVector_P1[2 * readno + 1].score)
                {
                    outputBestScoreVector.push_back(bestScoreVector_P1[2 * readno]);
                    outputBestScoreVector[readno].strand = '+';

                    std::string read_reverse(readSet_P1[2 * readno]);
                    hdga::seqUtils::reverse(readSet_P1[2 * readno], read_reverse);
                    readSet_P2.push_back(read_reverse);
                }
                else
                {
                    outputBestScoreVector.push_back(bestScoreVector_P1[2 * readno + 1]);
                    outputBestScoreVector[readno].strand = '-';

                    std::string read_reverse(readSet_P1[2 * readno + 1]);
                    hdga::seqUtils::reverse(readSet_P1[2 * readno + 1], read_reverse);
                    readSet_P2.push_back(read_reverse);
                }

                if (globalQryIds == nullptr)
                    outputBestScoreVector[readno].qryId = readno;
                else
                    outputBestScoreVector[readno].qryId = (*globalQryIds)[readno];

                if (readSet[readno].length() > maxReadLength)
                    maxReadLength = readSet[readno].length();
            }

            assert(outputBestScoreVector.size() == readSet.size());
            assert(readSet_P2.size() == readSet.size());

            // align reverse read to ref.
#if defined(USE_GPU_PHASE2)
            // if (useGpuPhase2)
            // {
            //     t2.set_method("GPU backward search");
            //     Phase2_Vectorized_GPU<int8_t> aligner_Phase2(readSet_P2, graph, parameters);
            //     aligner_Phase2.alignToDAGLocal_Phase2_vectorized_wrapper(outputBestScoreVector);

            //     if (parameters.banded)
            //     {
            //         PhaseTimer t2_2("traceback", 1, "vectorized banded");
            //         Vectorized_Banded<SimdInst<int32_t>> traceback_aligner(readSet_P2, graph, parameters);
            //         std::vector<int32_t> R(graph.numVertices);
            //         traceback_aligner.CalculateR(R);
            //         traceback_aligner.traceback_poa_avx2(R, outputBestScoreVector);
            //     }
            //     else
            //     {
            //         PhaseTimer t2_2("traceback", 1, "disable banded");
            //         NonBanded traceback_aligner(readSet_P2, graph, parameters);
            //         std::vector<std::string> readSet_traceback;
            //         for (size_t readno = 0; readno < readSet_P2.size(); readno++)
            //         {
            //             std::string read_reverse(readSet_P2[readno]);
            //             hdga::seqUtils::reverse(readSet_P2[readno], read_reverse);
            //             readSet_traceback.push_back(read_reverse);
            //         }
            //         traceback_aligner.traceback_nonbanded(readSet_traceback, outputBestScoreVector);
            //     }
            // }
            // else
            {
                if (parameters.banded)
                {
                    // use vectorized adaptive banded alignment for phase 2
                    t2.set_method("vectorized banded");
                    Vectorized_Banded<SimdInst<int32_t>> aligner_Phase2(readSet_P2, graph, parameters);
                    aligner_Phase2.Backward_and_Traceback_Vectorized_Banded(outputBestScoreVector);
                }
                else
                {
                    // use scalar non-banded alignment for phase 2
                    t2.set_method("disable banded");
                    NonBanded aligner_Phase2(readSet_P2, graph, parameters);
                    aligner_Phase2.Backward_and_Traceback_NonBanded(outputBestScoreVector);
                }
            }
#else
            (void)useGpuPhase2;
            if (parameters.banded)
            {
                // use vectorized adaptive banded alignment for phase 2
                t2.set_method("vectorized banded");
                Vectorized_Banded<SimdInst<int32_t>> aligner_Phase2(readSet_P2, graph, parameters);
                aligner_Phase2.Backward_and_Traceback_Vectorized_Banded(outputBestScoreVector);
            }
            else
            {
                // use scalar non-banded alignment for phase 2
                t2.set_method("disable banded");
                NonBanded aligner_Phase2(readSet_P2, graph, parameters);
                aligner_Phase2.Backward_and_Traceback_NonBanded(outputBestScoreVector);
            }
#endif
        }

#ifdef DEBUG
        {
            for (size_t readno = 0; readno < readSet.size(); readno++)
            {
                std::cout << "INFO, hdga::alignToDAGLocal, read # " << readno + 1 << ", score = " << outputBestScoreVector[readno].score
                          << ", backward score = " << outputBestScoreVector[readno].backward_score
                          << ", refColumnStart = " << outputBestScoreVector[readno].refColumnStart
                          << ", refColumnEnd = " << outputBestScoreVector[readno].refColumnEnd
                          << ", qryRowStart = " << outputBestScoreVector[readno].qryRowStart
                          << ", qryRowEnd = " << outputBestScoreVector[readno].qryRowEnd << "\n";
            }
        }
#endif
    }

    void alignToDAGLocal(const std::vector<std::string> &readSet,
                         const CSR_char_container &graph,
                         const Parameters &parameters,
                         std::vector<BestScoreInfo> &outputBestScoreVector)
    {
        bool useGpuPhase1 = false;
        bool useGpuPhase2 = false;
#ifdef USE_GPU_PHASE1
        useGpuPhase1 = true;
#endif
#ifdef USE_GPU_PHASE2
        useGpuPhase2 = true;
#endif
        alignToDAGLocalWithBackend(readSet, graph, parameters, outputBestScoreVector, useGpuPhase1, useGpuPhase2, nullptr);
    }

#if defined(USE_GPU_PHASE1) || defined(USE_GPU_PHASE2)
    void alignToDAGLocalHybrid(const std::vector<std::string> &readSet,
                               const CSR_char_container &graph,
                               const Parameters &parameters,
                               std::vector<BestScoreInfo> &outputBestScoreVector)
    {
        const auto hybridStart = std::chrono::high_resolution_clock::now();
        assert(outputBestScoreVector.empty());
        if (readSet.empty())
            return;

        double cpuSpeed = parameters.cpuSpeed;
        double gpuSpeed = parameters.gpuSpeed;
        if (cpuSpeed <= 0)
            cpuSpeed = 1.0;
        if (gpuSpeed <= 0)
            gpuSpeed = 1.0;

        std::vector<size_t> sortedOrder(readSet.size());
        std::iota(sortedOrder.begin(), sortedOrder.end(), 0);
        std::stable_sort(sortedOrder.begin(), sortedOrder.end(), [&](const size_t a, const size_t b)
                         { return readSet[a].length() > readSet[b].length(); });

        std::vector<uint8_t> assignToGpu(readSet.size(), 0);
        double cpuLoad = 0.0;
        double gpuLoad = 0.0;
        for (const size_t idx : sortedOrder)
        {
            const double cost = static_cast<double>(readSet[idx].length());
            const double cpuFinish = (cpuLoad + cost) / cpuSpeed;
            const double gpuFinish = (gpuLoad + cost) / gpuSpeed;
            if (gpuFinish <= cpuFinish)
            {
                assignToGpu[idx] = 1;
                gpuLoad += cost;
            }
            else
            {
                cpuLoad += cost;
            }
        }

        size_t gpuCount = std::count(assignToGpu.begin(), assignToGpu.end(), 1);
        if (readSet.size() > 1 && gpuCount == 0)
        {
            assignToGpu[sortedOrder.front()] = 1;
            gpuCount = 1;
        }
        if (readSet.size() > 1 && gpuCount == readSet.size())
        {
            assignToGpu[sortedOrder.back()] = 0;
            gpuCount -= 1;
        }

        std::vector<std::string> gpuReads;
        std::vector<std::string> cpuReads;
        std::vector<int32_t> gpuGlobalIds;
        std::vector<int32_t> cpuGlobalIds;
        gpuReads.reserve(gpuCount);
        cpuReads.reserve(readSet.size() - gpuCount);
        gpuGlobalIds.reserve(gpuCount);
        cpuGlobalIds.reserve(readSet.size() - gpuCount);

        for (size_t i = 0; i < readSet.size(); i++)
        {
            if (assignToGpu[i])
            {
                gpuReads.push_back(readSet[i]);
                gpuGlobalIds.push_back((int32_t)i);
            }
            else
            {
                cpuReads.push_back(readSet[i]);
                cpuGlobalIds.push_back((int32_t)i);
            }
        }

        bool useGpuPhase1 = false;
        bool useGpuPhase2 = false;
#ifdef USE_GPU_PHASE1
        useGpuPhase1 = true;
#endif
#ifdef USE_GPU_PHASE2
        useGpuPhase2 = true;
#endif

        std::cout << "INFO, hdga::alignToDAGLocalHybrid, cpu reads = " << cpuReads.size()
                  << ", gpu reads = " << gpuReads.size()
                  << ", cpuSpeed = " << cpuSpeed
                  << ", gpuSpeed = " << gpuSpeed << std::endl;

        std::vector<BestScoreInfo> gpuOutput;
        std::vector<BestScoreInfo> cpuOutput;
        double cpuElapsedSec = 0.0;
        double gpuElapsedSec = 0.0;

        std::thread gpuThread;
        std::thread cpuThread;

        if (!gpuReads.empty())
        {
            gpuThread = std::thread([&]()
                                    {
                                        const auto t0 = std::chrono::high_resolution_clock::now();
                                        alignToDAGLocalWithBackend(gpuReads, graph, parameters, gpuOutput, useGpuPhase1, useGpuPhase2, &gpuGlobalIds);
                                        const auto t1 = std::chrono::high_resolution_clock::now();
                                        gpuElapsedSec = std::chrono::duration<double>(t1 - t0).count(); });
        }
        if (!cpuReads.empty())
        {
            cpuThread = std::thread([&]()
                                    {
                                        const auto t0 = std::chrono::high_resolution_clock::now();
                                        alignToDAGLocalWithBackend(cpuReads, graph, parameters, cpuOutput, false, false, &cpuGlobalIds);
                                        const auto t1 = std::chrono::high_resolution_clock::now();
                                        cpuElapsedSec = std::chrono::duration<double>(t1 - t0).count(); });
        }

        if (gpuThread.joinable())
            gpuThread.join();
        if (cpuThread.joinable())
            cpuThread.join();

        outputBestScoreVector.resize(readSet.size());
        for (auto &entry : gpuOutput)
            outputBestScoreVector[entry.qryId] = std::move(entry);
        for (auto &entry : cpuOutput)
            outputBestScoreVector[entry.qryId] = std::move(entry);

        const auto hybridEnd = std::chrono::high_resolution_clock::now();
        const double hybridWallSec = std::chrono::duration<double>(hybridEnd - hybridStart).count();
        const double maxDeviceSec = std::max(cpuElapsedSec, gpuElapsedSec);
        std::cout << "INFO, hdga::alignToDAGLocalHybrid, cpu_time = " << cpuElapsedSec
                  << " s, gpu_time = " << gpuElapsedSec
                  << " s, max_time = " << maxDeviceSec
                  << " s, hybrid_total = " << hybridWallSec << " s" << std::endl;
    }
#endif

    /**
     * @brief                                 alignment routine
     * @param[in]   reads                     vector of strings
     * @param[in]   graph
     * @param[in]   parameters                input parameters
     * @param[in]   mode                      alignment mode
     * @param[out]  outputBestScoreVector
     */
    void alignToDAG(const std::vector<std::string> &reads,
                    const CSR_char_container &graph,
                    const Parameters &parameters,
                    const MODE mode,
                    std::vector<BestScoreInfo> &outputBestScoreVector)
    {
        // TODO: Support other alignment modes: global and semi-global
        switch (mode)
        {
        case LOCAL:
            if (parameters.hybridCpuGpu)
            {
#if defined(USE_GPU_PHASE1) || defined(USE_GPU_PHASE2)
                alignToDAGLocalHybrid(reads, graph, parameters, outputBestScoreVector);
#else
                std::cout << "WARN, hdga::alignToDAG, hybrid mode requested but GPU backend is unavailable in this binary; using CPU mode" << std::endl;
                alignToDAGLocal(reads, graph, parameters, outputBestScoreVector);
#endif
            }
            else
            {
                alignToDAGLocal(reads, graph, parameters, outputBestScoreVector);
            }
            break;
        default:
            std::cerr << "ERROR, hdga::alignToDAG, Invalid alignment mode";
            exit(1);
        }
    }

    /**
     * @brief                                 print alignment results to file
     * @param[in]   parameters                input parameters
     * @param[in]   graph
     * @param[in]   outputBestScoreVector
     */
    void printResultsToFile(const Parameters &parameters,
                            const std::vector<ContigInfo> &qmetadata,
                            const CSR_char_container &graph,
                            const std::vector<BestScoreInfo> &outputBestScoreVector)
    {
        std::ofstream outstrm(parameters.ofile);

        assert(qmetadata.size() == outputBestScoreVector.size());

        for (auto &e : outputBestScoreVector)
        {
            outstrm << qmetadata[e.qryId].name << "\t"
                    << qmetadata[e.qryId].len << "\t"
                    << e.qryRowStart << "\t"
                    << e.qryRowEnd << "\t"
                    << e.refColumnStart << "\t"
                    << e.refColumnEnd << "\t"
                    << e.strand << "\t"
                    << graph.originalVertexId[e.refColumnStart] << "\t"
                    << graph.originalVertexId[e.refColumnEnd] << "\t"
                    << e.score << "\t"
                    << e.backward_score << "\t"
                    << e.cigar << "\n";
        }
    }

    /**
     * @brief                                 alignment routine
     * @param[in]   parameters                input parameters
     * @param[in]   mode                      alignment mode
     * @param[out]  outputBestScoreVector
     */
    int alignToDAG(const Parameters &parameters,
                   const MODE mode,
                   std::vector<BestScoreInfo> &outputBestScoreVector)
    {
        hdga::graphLoader g;

        // Parse all reads into a vector
        std::vector<std::string> reads;
        assert(outputBestScoreVector.empty());

        // read metadata
        std::vector<ContigInfo> qmetadata;

        {
            if (parameters.mode.compare("vg") == 0)
                g.loadFromVG(parameters.rfile);
            else if (parameters.mode.compare("txt") == 0)
                g.loadFromTxt(parameters.rfile);
            else
            {
                std::cerr << "Invalid format " << parameters.mode << std::endl;
                exit(1);
            }
        }

        {
            // TODO: Read query sequences in batches rather than all at once
            if (!fileExists(parameters.qfile))
            {
                std::cerr << parameters.qfile << " not accessible." << std::endl;
                exit(1);
            }

            // Open the file using kseq
            FILE *file = fopen(parameters.qfile.c_str(), "r");
            assert(file != NULL);
            gzFile fp = gzdopen(fileno(file), "r");
            kseq_t *seq = kseq_init(fp);

            // size of sequence
            int len;

            while ((len = kseq_read(seq)) >= 0)
            {
                hdga::seqUtils::makeUpperCase(seq->seq.s, len);
                reads.push_back(seq->seq.s);

                // record query name and length
                qmetadata.push_back(ContigInfo{seq->name.s, (int32_t)seq->seq.l});
            }

            // Close the input file
            kseq_destroy(seq);
            gzclose(fp);
            fclose(file);
        }

        alignToDAG(reads, g.diCharGraph, parameters, mode, outputBestScoreVector);

        // print results
        printResultsToFile(parameters, qmetadata, g.diCharGraph, outputBestScoreVector);

        return hdga_STATUS_OK;
    }
}

#endif
