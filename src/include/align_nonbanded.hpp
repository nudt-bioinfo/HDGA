/**
 * @file    align_vectorized_banded.hpp
 * @brief   vectorized DP computation using adaptive banded strategy
 * @author  Qingzhe Wang <1197460504@qq.com>
 */

#ifndef GRAPH_ALIGN_NONBANDED_HPP
#define GRAPH_ALIGN_NONBANDED_HPP

#include <immintrin.h>
#include <x86intrin.h>

#include "graphLoad.hpp"
#include "csr_char.hpp"
#include "graph_iter.hpp"
#include "base_types.hpp"
#include "utils.hpp"
#include <queue>

namespace dga
{
    /**
     * @brief  vectorized banded alignment in reverse phase
     */
    class NonBanded
    {
    private:
        // reference graph
        const CSR_char_container &graph;

        // pre-compute which graph vertices are connected with hop longer
        // than 'blockWidth'
        std::vector<bool> withLongHop;

        // input reads
        const std::vector<std::string> &readSet;

        // input parameters (e.g., scoring scheme)
        const Parameters &parameters;

        // maximum bandwidth (default value)
        const int MAX_BANDWIDTH = 1024;

        struct DPRow
        {
            int Bstart, Bend;   // band range in this row
            int max_index;      // index of maximum value in this row
            int max_index_left; // leftmost index of maximum value in this row
            int max_value;      // maximum value in this row
            std::vector<int32_t, aligned_alloc<int32_t, 64>> values;
        };

    public:
        // number of previous rows to be stored
        static constexpr size_t MAX_PREV_ROWS = 8;

        /**
         * @brief                   public constructor
         * @param[in]   readSet     vector of input query sequences to align
         * @param[in]   g           input reference graph
         */
        NonBanded(const std::vector<std::string> &readSet,
                  const CSR_char_container &g,
                  const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->computeLongHops();
        };

        void Backward_and_Traceback_NonBanded(std::vector<BestScoreInfo> &bestScoreVector)
        {
            assert(bestScoreVector.size() == readSet.size());

            // [Phase 2.1] backward computing (disable banded)
            {
                PhaseTimer t2_1("backward computing", 1, "disable banded");
                backward_nonbanded(bestScoreVector);
            }

            // [Phase 2.2] traceback (disable banded)
            {
                PhaseTimer t2_2("traceback", 1, "disable banded");
                std::vector<std::string> readSet_traceback; // input reads of traceback stage
                for (size_t readno = 0; readno < readSet.size(); readno++)
                {
                    std::string read_reverse(readSet[readno]);
                    dga::seqUtils::reverse(readSet[readno], read_reverse);
                    readSet_traceback.push_back(read_reverse);
                }
                traceback_nonbanded(readSet_traceback, bestScoreVector);
            }
        }

    private:
        /**
         * @brief     precompute which vertices are connected with long edge hops
         *            i.e. longer than 'blockWidth'
         */
        void computeLongHops()
        {
            assert(withLongHop.size() == 0);

            this->withLongHop.resize(graph.numVertices, false);

            for (int32_t i = 0; i < graph.numVertices; i++)
            {
                for (auto j = graph.offsets_in[i]; j < graph.offsets_in[i + 1]; j++)
                {
                    auto from_pos = graph.adjcny_in[j];
                    auto to_pos = i;

                    // compare hop distance to 'blockWidth'
                    if (to_pos - from_pos >= this->MAX_PREV_ROWS)
                        this->withLongHop[to_pos] = true;
                }
            }

#ifdef DEBUG
            auto trueCount = std::count(withLongHop.begin(), withLongHop.end(), true);
            std::cout << "INFO, dga::Phase1_Rev_Vectorized_Banded::computeLongHops, fraction of hops that are long: " << trueCount * 1.0 / graph.numVertices << "\n";
#endif
        }

        /**
         * @brief the scalar non-banded alignment
         * @param bestScoreVector
         */
        void backward_nonbanded(std::vector<BestScoreInfo> &bestScoreVector)
        {
            // copy graph as function variable for faster access
            const CSR_char_container graphLocal = this->graph;
            const std::vector<bool> withLongHopLocal = withLongHop;

            // the maximum read length
            int32_t max_readLen = INT32_MIN;
            for (size_t readno = 0; readno < readSet.size(); readno++)
            {
                max_readLen = std::max(max_readLen, bestScoreVector[readno].qryRowEnd + 1);
            }

#pragma omp parallel
            {
                // store farther rows in buffer
                std::size_t countLongHops = std::count(withLongHopLocal.begin(), withLongHopLocal.end(), true);
                std::vector<DPRow> fartherRowsBuffer(countLongHops);
                for (std::size_t i = 0; i < countLongHops; i++)
                {
                    fartherRowsBuffer[i].Bstart = 0;
                    fartherRowsBuffer[i].Bend = 0;
                    fartherRowsBuffer[i].values.resize(max_readLen, 0);
                }
                std::vector<DPRow *> fartherRows(graphLocal.numVertices);
                {
                    size_t j = 0;
                    for (int32_t i = 0; i < graphLocal.numVertices; i++)
                        if (this->withLongHop[i])
                            fartherRows[i] = &fartherRowsBuffer[j++];
                }

                // loop buffer to store cells in near predecessor rows
                DPRow dp_buffer[MAX_PREV_ROWS];
                for (int i = 0; i < MAX_PREV_ROWS; i++)
                {
                    dp_buffer[i].Bstart = 0;
                    dp_buffer[i].Bend = 0;
                    dp_buffer[i].values.resize(max_readLen, 0);
                }

#pragma omp for schedule(dynamic)
                for (size_t readno = 0; readno < readSet.size(); readno++)
                {
                    const std::string &read = readSet[readno];                 // current read
                    int32_t qryEnd = bestScoreVector[readno].qryRowEnd;        // ending position of query read
                    int32_t refEnd = bestScoreVector[readno].refColumnEnd;     // ending position of reference graph
                    int32_t forward_bestScore = bestScoreVector[readno].score; // bestscore in the forward computing phase
                    int readLen = qryEnd + 1;                                  // readlen from 0 to ending postion
                    int Rstart = read.length() - qryEnd - 1;                   // starting index of reversed read
                    int32_t bestScore = 0;
                    int32_t bestRow = 0, bestCol = 0;

                    /* reverse DP procedure to locate starting position */
                    bool found = false;
                    for (int32_t i = refEnd; i >= 0; i--)
                    {
                        DPRow *current_row = &dp_buffer[i % MAX_PREV_ROWS]; // the pointer of current row in the dp_buffer
                        int current_row_max = INT32_MIN;                    // maximum of current row
                        int current_row_max_index;                          // index of maximum of current row

                        // current graph char
                        char graphChar = graphLocal.vertex_label[i];

                        /* 1. computing in current row, the read has been reversed */
                        for (int32_t j = 0; j < readLen; j++)
                        {
                            char readChar = read[Rstart + j];

                            int32_t currentMax = 0;

                            // match/mismatch score
                            int32_t matchScore = readChar == graphChar ? parameters.match : -1 * parameters.mismatch;
                            currentMax = dga_max(currentMax, matchScore); // local alignment can also start with a match at this char

                            // match & delete score
                            for (auto k = graphLocal.offsets_out[i]; k < graphLocal.offsets_out[i + 1]; k++)
                            {
                                size_t m = graphLocal.adjcny_out[k]; // index of predecessor row in reference graph
                                DPRow *prow;                         // pointer to the predecessor row
                                int distance = m - i;                // distance between current and predecessor rows
                                if (m > refEnd)
                                    continue;
                                if (distance >= MAX_PREV_ROWS)
                                    prow = fartherRows[m];
                                else
                                    prow = &dp_buffer[m % MAX_PREV_ROWS];

                                // delete score
                                currentMax = dga_max(currentMax, prow->values[j] - parameters.del);

                                // match score
                                currentMax = dga_max(currentMax, prow->values[j - 1] + matchScore);
                            }

                            // insert score
                            if (j > 0)
                                currentMax = dga_max(currentMax, current_row->values[j - 1] - parameters.ins);

                            current_row->values[j] = currentMax;

                            if (currentMax >= bestScore)
                            {
                                bestCol = i;
                                bestRow = j;
                                bestScore = currentMax;
                            }
                            if (bestScore == forward_bestScore)
                            {
                                found = true;
                                break;
                            }
                        }

                        /* 2. break if have found bestScore */
                        if (found)
                            break;

                        /* 3. update fartherRowsbuffer if current graph node is longHop node */
                        if (withLongHopLocal[i])
                        {
                            for (int32_t j = 0; j < readLen; j++)
                                fartherRows[i]->values[j] = current_row->values[j];
                        }
                    } /* DP matrix complete */

                    bestScoreVector[readno].refColumnStart = bestCol;
                    bestScoreVector[readno].qryRowStart = read.length() - 1 - (Rstart + bestRow);
                    bestScoreVector[readno].backward_score = bestScore;
                    if (bestScore != forward_bestScore)
                        fprintf(stderr, "readno: #%d, forward_score = %d, backward_score = %d\n", readno + 1, forward_bestScore, bestScore);
                }
            }
        }

        /**
         * @brief                         execute second phase of alignment i.e. compute cigar
         * @param[in]   readSet
         * @param[in]   bestScoreVector   best score and alignment location for each read
         */
        void
        traceback_nonbanded(const std::vector<std::string> &readSet,
                            std::vector<BestScoreInfo> &bestScoreVector)
        {
            assert(bestScoreVector.size() == readSet.size());

#pragma omp parallel
            {
#pragma omp for schedule(dynamic) nowait
                for (size_t readno = 0; readno < readSet.size(); readno++)
                {
                    // for time profiling within phase 2
                    uint64_t time_p2_1, time_p2_2;

                    // read length
                    auto readLength = readSet[readno].length();

                    //
                    // PHASE 2.1 : RECOMPUTE DP MATRIX WITH TRACEBACK INFORMATION
                    // recomputation is done within selected block of DP matrix
                    //

                    // width of score matrix that we need in memory
                    std::size_t reducedWidth = bestScoreVector[readno].refColumnEnd - bestScoreVector[readno].refColumnStart + 1;

                    // for new beginning column
                    std::size_t j0 = bestScoreVector[readno].refColumnStart;

                    // height of scoring matrix for re-computation
                    std::size_t reducedHeight = bestScoreVector[readno].qryRowEnd - bestScoreVector[readno].qryRowStart + 1;

                    // for new beginning row
                    std::size_t i0 = bestScoreVector[readno].qryRowStart;

                    // scores in the last row
                    std::vector<int32_t> finalRow(reducedWidth, 0);

                    // complete score matrix of size height x width to allow traceback
                    // Note: to optimize storge, we only store vertical difference; absolute values of
                    //       which is bounded by gap penalty
#ifdef DEBUG
                    std::cout << "INFO, dga::alignToDAGLocal_Phase2, aligning read #" << readno + 1 << ", memory requested= " << reducedWidth * reducedHeight << " bytes" << std::endl;
#endif

                    std::vector<std::vector<int8_t>> completeMatrixLog(reducedHeight, std::vector<int8_t>(reducedWidth, 0));
                    std::vector<std::vector<int8_t>> Log(reducedHeight, std::vector<int8_t>(reducedWidth, 0));

                    {
                        auto tick1 = __rdtsc();

                        // scoring matrix of size 2 x width, init with zero
                        std::vector<std::vector<int32_t>> matrix(2, std::vector<int32_t>(reducedWidth, 0));

                        // iterate over characters in read
                        for (std::size_t i = 0; i < reducedHeight; i++)
                        {
                            // iterate over characters in reference graph
                            for (std::size_t j = 0; j < reducedWidth; j++)
                            {
                                // current reference character
                                char curChar = graph.vertex_label[j + j0];

                                // insertion edit
                                int32_t fromInsertion = matrix[(i - 1) & 1][j] - parameters.ins;
                                //'& 1' is same as doing modulo 2

                                // match-mismatch edit
                                int32_t matchScore = curChar == readSet[readno][i + i0] ? parameters.match : -1 * parameters.mismatch;
                                int32_t fromMatch = matchScore; // also handles the case when in-degree is zero

                                // deletion edit
                                int32_t fromDeletion = -1;

                                for (auto k = graph.offsets_in[j + j0]; k < graph.offsets_in[j + j0 + 1]; k++)
                                {
                                    // ignore edges outside the range
                                    if (graph.adjcny_in[k] >= j0)
                                    {
                                        fromMatch = dga_max(fromMatch, matrix[(i - 1) & 1][graph.adjcny_in[k] - j0] + matchScore);
                                        fromDeletion = dga_max(fromDeletion, matrix[i & 1][graph.adjcny_in[k] - j0] - parameters.del);
                                    }
                                }

                                // evaluate current score
                                matrix[i & 1][j] = dga_max(dga_max(fromInsertion, fromMatch), dga_max(fromDeletion, 0));

                                // save vertical difference of scores, used later for backtracking
                                completeMatrixLog[i][j] = matrix[i & 1][j] - matrix[(i - 1) & 1][j];
                            }

                            // Save last row
                            if (i == reducedHeight - 1)
                                finalRow = matrix[i & 1];
                        }

                        int32_t bestScoreReComputed = *std::max_element(finalRow.begin(), finalRow.end());

                        // the recomputed score and its location should match our original calculation
                        assert(bestScoreReComputed == bestScoreVector[readno].score);
                        assert(bestScoreReComputed == finalRow[bestScoreVector[readno].refColumnEnd - j0]);

                        auto tick2 = __rdtsc();
                        time_p2_1 = tick2 - tick1;
                    }

                    //
                    // PHASE 2.2 : COMPUTE CIGAR
                    //

                    std::string cigar;

                    {
                        auto tick1 = __rdtsc();

                        std::vector<int32_t> currentRowScores = finalRow;
                        std::vector<int32_t> aboveRowScores(reducedWidth);

                        int col = reducedWidth - 1;
                        int row = reducedHeight - 1;

                        while (col >= 0 && row >= 0)
                        {
                            Log[row][col] = 1;

                            if (currentRowScores[col] <= 0)
                                break;

                            // retrieve score values from vertical score differences
                            for (std::size_t i = 0; i < reducedWidth; i++)
                                aboveRowScores[i] = currentRowScores[i] - completeMatrixLog[row][i];

                            // current reference character
                            char curChar = graph.vertex_label[col + j0];

                            // insertion edit
                            int32_t fromInsertion = aboveRowScores[col] - parameters.ins;

                            // match-mismatch edit
                            int32_t matchScore = curChar == readSet[readno][row + i0] ? parameters.match : -1 * parameters.mismatch;

                            int32_t fromMatch = matchScore; // also handles the case when in-degree is zero
                            std::size_t fromMatchPos = col;

                            // deletion edit
                            int32_t fromDeletion = -1;
                            std::size_t fromDeletionPos;

                            for (auto k = graph.offsets_in[col + j0]; k < graph.offsets_in[col + j0 + 1]; k++)
                            {
                                if (graph.adjcny_in[k] >= j0)
                                {
                                    auto fromCol = graph.adjcny_in[k] - j0;

                                    if (fromMatch < aboveRowScores[fromCol] + matchScore)
                                    {
                                        fromMatch = aboveRowScores[fromCol] + matchScore;
                                        fromMatchPos = fromCol;
                                    }

                                    if (fromDeletion < currentRowScores[fromCol] - parameters.del)
                                    {
                                        fromDeletion = currentRowScores[fromCol] - parameters.del;
                                        fromDeletionPos = fromCol;
                                    }
                                }
                            }

                            // evaluate recurrence
                            {
                                if (currentRowScores[col] == fromMatch)
                                {
                                    if (matchScore == parameters.match)
                                        cigar.push_back('=');
                                    else
                                        cigar.push_back('X');

                                    // if alignment starts from this column, stop
                                    if (fromMatchPos == col)
                                        break;

                                    // shift to preceeding column
                                    col = fromMatchPos;

                                    // shift to above row
                                    row--;
                                    currentRowScores = aboveRowScores;
                                }
                                else if (currentRowScores[col] == fromDeletion)
                                {
                                    cigar.push_back('D');

                                    // shift to preceeding column
                                    col = fromDeletionPos;
                                }
                                else
                                {
                                    assert(currentRowScores[col] == fromInsertion);

                                    cigar.push_back('I');

                                    // shift to above row
                                    row--;
                                    currentRowScores = aboveRowScores;
                                }
                            }
                        }

                        // string reverse
                        std::reverse(cigar.begin(), cigar.end());

                        // shorten the cigar string
                        dga::seqUtils::cigarCompact(cigar);

                        // validate if cigar yields best score
                        assert(dga::seqUtils::cigarScore(cigar, parameters) == bestScoreVector[readno].score);

                        bestScoreVector[readno].cigar = cigar;

                        auto tick2 = __rdtsc();
                        time_p2_2 = tick2 - tick1;
                    }

#ifdef DEBUG
                    std::cout << "INFO, dga::alignToDAGLocal_Phase2, aligning read #" << readno + 1 << ", len = " << readLength << ", score " << bestScoreVector[readno].score << ", strand " << bestScoreVector[readno].strand << "\n";
                    std::cout << "INFO, dga::alignToDAGLocal_Phase2, cigar: " << bestScoreVector[readno].cigar << "\n";
                    std::cout << "TIMER, dga::alignToDAGLocal_Phase2, CPU cycles spent in :  phase 2.1 = " << time_p2_1 * 1.0 / ASSUMED_CPU_FREQ << ", phase 2.2 = " << time_p2_2 * 1.0 / ASSUMED_CPU_FREQ << "\n";
                    // std::cout.flush();
#endif
                }
            }
        }
    };

} // namespace dga
#endif