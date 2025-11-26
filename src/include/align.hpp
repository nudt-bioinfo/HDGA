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
#include <fstream>
#include <sstream>
#include <chrono>

// External includes
#include "kseq/kseq.h"

KSEQ_INIT(gzFile, gzread)

#define dga_max(a, b) (((a) > (b)) ? (a) : (b))

namespace dga
{
    /**
     * @brief                               local alignment routine
     * @param[in]   readSet
     * @param[in]   graph                   node-labeled directed graph
     * @param[in]   parameters              input parameters
     * @param[out]  outputBestScoreVector
     */
    void alignToDAGLocal(const std::vector<std::string> &readSet,
                         const CSR_char_container &graph,
                         const Parameters &parameters,
                         std::vector<BestScoreInfo> &outputBestScoreVector)
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
                dga::seqUtils::reverseComplement(readSet[readno], read_reverse);

                readSet_P1.push_back(readSet[readno]);
                readSet_P1.push_back(read_reverse);

                if (readSet[readno].length() > maxReadLength)
                    maxReadLength = readSet[readno].length();
            }

            assert(bestScoreVector_P1.size() == 2 * readSet.size());
            assert(readSet_P1.size() == 2 * readSet.size());

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

#ifdef DEBUG
        {
            for (size_t readno = 0; readno < readSet_P1.size(); readno++)
            {
                std::cout << "INFO, dga::alignToDAGLocal, read # " << readno + 1 << ", score = " << bestScoreVector_P1[readno].score
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
                    dga::seqUtils::reverse(readSet_P1[2 * readno], read_reverse);
                    readSet_P2.push_back(read_reverse);
                }
                else
                {
                    outputBestScoreVector.push_back(bestScoreVector_P1[2 * readno + 1]);
                    outputBestScoreVector[readno].strand = '-';

                    std::string read_reverse(readSet_P1[2 * readno + 1]);
                    dga::seqUtils::reverse(readSet_P1[2 * readno + 1], read_reverse);
                    readSet_P2.push_back(read_reverse);
                }

                outputBestScoreVector[readno].qryId = readno;

                if (readSet[readno].length() > maxReadLength)
                    maxReadLength = readSet[readno].length();
            }

            assert(outputBestScoreVector.size() == readSet.size());
            assert(readSet_P2.size() == readSet.size());

            // align reverse read to ref.
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

#ifdef DEBUG
        {
            for (size_t readno = 0; readno < readSet.size(); readno++)
            {
                std::cout << "INFO, dga::alignToDAGLocal, read # " << readno + 1 << ", score = " << outputBestScoreVector[readno].score
                          << ", backward score = " << outputBestScoreVector[readno].backward_score
                          << ", refColumnStart = " << outputBestScoreVector[readno].refColumnStart
                          << ", refColumnEnd = " << outputBestScoreVector[readno].refColumnEnd
                          << ", qryRowStart = " << outputBestScoreVector[readno].qryRowStart
                          << ", qryRowEnd = " << outputBestScoreVector[readno].qryRowEnd << "\n";
            }
        }
#endif
    }

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
            alignToDAGLocal(reads, graph, parameters, outputBestScoreVector);
            break;
        default:
            std::cerr << "ERROR, dga::alignToDAG, Invalid alignment mode";
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
        dga::graphLoader g;

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
                dga::seqUtils::makeUpperCase(seq->seq.s, len);
                reads.push_back(seq->seq.s);

                // record query name and length
                qmetadata.push_back(ContigInfo{seq->name.s, (int32_t)seq->seq.l});
            }

            // Close the input file
            kseq_destroy(seq);
            gzclose(fp);
            fclose(file);
        }

        std::cout << "INFO, dga::alignToDAG, total count of reads = " << reads.size() << std::endl;

        alignToDAG(reads, g.diCharGraph, parameters, mode, outputBestScoreVector);

        // print results
        printResultsToFile(parameters, qmetadata, g.diCharGraph, outputBestScoreVector);

        return dga_STATUS_OK;
    }
}

#endif
