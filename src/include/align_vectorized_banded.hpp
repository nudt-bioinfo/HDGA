/**
 * @file    align_vectorized_banded.hpp
 * @brief   vectorized DP computation using adaptive banded strategy
 * @author  Qingzhe Wang <1197460504@qq.com>
 */

#ifndef GRAPH_ALIGN_VEC_BANDED_HPP
#define GRAPH_ALIGN_VEC_BANDED_HPP

#include <immintrin.h>
#include <x86intrin.h>

#include "graphLoad.hpp"
#include "csr_char.hpp"
#include "graph_iter.hpp"
#include "base_types.hpp"
#include "utils.hpp"
#include <queue>

// 引用正向的向量化差分头文件，获取封装好的向量操作
#include "align_vectorized_difference.hpp"

namespace dga
{

    /**
     * @brief  vectorized banded alignment in reverse phase
     */
    template <typename SIMD>
    class Vectorized_Banded
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

        const double b = 10;   // banded parameter
        const double f = 0.01; // banded parameter

        struct DPRow
        {
            int Bstart, Bend;   // band range in this row
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
        Vectorized_Banded(const std::vector<std::string> &readSet,
                          const CSR_char_container &g,
                          const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->computeLongHops();
        };

        void Backward_and_Traceback_Vectorized_Banded(std::vector<BestScoreInfo> &bestScoreVector)
        {
            assert(bestScoreVector.size() == readSet.size());

            // R value of each vertex in the reference graph
            std::vector<int32_t> R(graph.numVertices);
            // calculate R value of reference graph
            CalculateR(R);

            // [Phase 2.1] backward computing using vectorized banded strategy
            {
                PhaseTimer t2_1("backward computing", 1, "vectorized banded");
                backward_poa_avx2(R, bestScoreVector);
            }
            // [Phase 2.2] traceback using vectorized banded strategy
            {
                PhaseTimer t2_2("traceback", 1, "vectorized banded");
                traceback_poa_avx2(R, bestScoreVector);
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
         * @brief calculate Rvalue of the reference graph
         *
         * @param graph reference graph
         * @param R R value of each vertex of the graph
         */
        void CalculateR(std::vector<int32_t> &R)
        {
            // indegree of each vertex in the graph
            std::vector<int> D(graph.numVertices);

            for (size_t i = 0; i < graph.numVertices; i++)
            {
                R[i] = i + 1;
                D[i] = graph.offsets_in[i + 1] - graph.offsets_in[i];
            }

            /* std::queue<int> Q;
            Q.push(0);
            while (!Q.empty())
            {
                int cur = Q.front();
                Q.pop();

                int Wmax = -1;
                int Sucmax = -1;
                for (int i = graph.offsets_in[cur]; i < graph.offsets_in[cur + 1]; i++)
                {
                    int pre = graph.adjcny_in[i];
                }

                if (cur == graph.numVertices - 1)
                    break;

                for (int i = graph.offsets_out[cur]; i < graph.offsets_out[cur + 1]; i++)
                {
                    int suc = graph.adjcny_out[i];
                    D[suc] = D[suc] - 1;
                    if (D[suc] == 0)
                        Q.push(suc);
                }
            } */
        }

        /**
         * @brief backward computing based on poa algorithm using avx2 simd instructions
         * @param R R value of each vertex of the graph
         * @param bestScoreVector
         */
        void backward_poa_avx2(const std::vector<int32_t> &R,
                               std::vector<BestScoreInfo> &bestScoreVector)
        {
            // number of cells processed in each vector operation
            int step = SIMD::numSeqs;

            // vectorized scoring parameters
            __mxxxi match = SIMD::set1((typename SIMD::type)parameters.match);
            __mxxxi mismatch = SIMD::set1((typename SIMD::type) - 1 * parameters.mismatch);
            __mxxxi del = SIMD::set1((typename SIMD::type) - 1 * parameters.del);
            __mxxxi ins = SIMD::set1((typename SIMD::type) - 1 * parameters.ins);

            // copy graph as function variable for faster access
            const CSR_char_container graphLocal = this->graph;
            const std::vector<bool> withLongHopLocal = withLongHop;

#pragma omp parallel
            {

#pragma omp for schedule(dynamic)
                for (size_t readno = 0; readno < readSet.size(); readno++)
                {
                    const std::string &read = readSet[readno];                 // current read
                    int32_t qryEnd = bestScoreVector[readno].qryRowEnd;        // ending position of current read
                    int32_t refEnd = bestScoreVector[readno].refColumnEnd;     // ending position of reference graph
                    int32_t forward_bestScore = bestScoreVector[readno].score; // bestScore in forward computing phase
                    int readLen = qryEnd + 1;                                  // effective read length
                    int refLen = refEnd + 1;                                   // effective reference graph length
                    int maxCol = 0;                                            // position of maximum score in previous row
                    int Mstart, Mend;                                          // maximum row range
                    int Pleft, Pright;                                         // the leftmost and rightmost positions of maximum values of all predecessor rows
                    int Bstart, Bend;                                          // band range in current row
                    int Rvalue;                                                // Rvalue of current vertex
                    int w = b + f * readLen;                                   // extra bases added to both sides of the band
                    int Rstart = read.length() - qryEnd - 1;                   // starting index of the reversed read
                    int32_t bestScore = 0;
                    int32_t bestRow = 0, bestCol = 0;

                    // store farther rows in buffer
                    std::size_t countLongHops = std::count(withLongHopLocal.begin(), withLongHopLocal.end(), true);
                    DPRow fartherRowsBuffer[countLongHops];
                    for (std::size_t i = 0; i < countLongHops; i++)
                    {
                        fartherRowsBuffer[i].Bstart = 0;
                        fartherRowsBuffer[i].Bend = 0;
                        fartherRowsBuffer[i].values.resize(MAX_BANDWIDTH, 0);
                    }
                    std::vector<DPRow *> fartherRows(graphLocal.numVertices);
                    {
                        size_t j = 0;
                        for (int32_t i = 0; i < graphLocal.numVertices; i++)
                            if (this->withLongHop[i])
                                fartherRows[i] = &fartherRowsBuffer[j++];
                    }

                    // loop buffer to store previous rows within MAX_PREV_ROWS
                    DPRow dp_buffer[MAX_PREV_ROWS];
                    for (int i = 0; i < MAX_PREV_ROWS; i++)
                    {
                        dp_buffer[i].Bstart = 0;
                        dp_buffer[i].Bend = 0;
                        dp_buffer[i].values.resize(MAX_BANDWIDTH, 0);
                    }

                    /* adaptive banded DP to locate starting position */
                    for (int32_t i = refEnd; i >= 0; i--)
                    {
                        /* 1. compute band range */
                        if (i == refEnd)
                            Mstart = Mend = 0;
                        else
                        {
                            Pleft = INT32_MAX;
                            Pright = INT32_MIN;
                            bool have_pre = false; // if there is any predecessor row
                            for (auto k = graphLocal.offsets_out[i]; k < graphLocal.offsets_out[i + 1]; k++)
                            {
                                size_t m = graphLocal.adjcny_out[k]; // index of predecessor row
                                DPRow *prow;                         // pointer to predecessor row
                                int distance = m - i;                // distance between the current row and predecessor
                                if (m > refEnd)
                                    continue;
                                have_pre = true;
                                if (distance >= MAX_PREV_ROWS)
                                    prow = fartherRows[m];
                                else
                                    prow = &dp_buffer[m % MAX_PREV_ROWS];

                                int pre_max_index_left = prow->max_index_left;
                                if (pre_max_index_left < Pleft)
                                    Pleft = pre_max_index_left;
                                if (pre_max_index_left > Pright)
                                    Pright = pre_max_index_left;
                            }
                            if (have_pre)
                            {
                                Mstart = Pleft + 1;
                                Mend = Pright + 1;
                            }
                            else
                                Mstart = Mend = 0;
                        }
                        Rvalue = R[i];
                        int Bstart0 = std::max(0, Mstart - w);
                        int Bend0 = std::min(readLen - 1, Mend + w);
                        Bstart = Bstart0 / step * step;
                        Bend = (Bend0 + step) / step * step - 1;
                        // the termination condition of DP
                        if (Bstart0 >= Bend0)
                            break;

                        DPRow *current_row = &dp_buffer[i % MAX_PREV_ROWS]; // the pointer to current row in dp_buffer
                        current_row->Bstart = Bstart;
                        current_row->Bend = Bend;
                        int current_row_max = INT32_MIN;                // maximum value in current row
                        int current_row_max_index_left;                 // index of maximum value in current row
                        std::vector<int> row_values(Bend - Bstart + 1); // store values within the band of current row

                        // current graph character
                        __mxxxi graphChar = SIMD::set1((typename SIMD::type)graphLocal.vertex_label[i]);
                        // read node corresponding to the current row
                        std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> readCharsInt(Bend - Bstart + 1);
                        for (int32_t j = Bstart; j <= Bend; j++)
                        {
                            if (j < readLen)
                                readCharsInt[j - Bstart] = read[Rstart + j];
                        }
                        if (Bend - Bstart + 1 > MAX_BANDWIDTH)
                        {
                            current_row->values.resize(Bend - Bstart + 1, 0);
                        }
                        // match & delete scores within the band
                        std::vector<int32_t, aligned_alloc<int32_t, 64>> match_delete(Bend - Bstart + 1);

                        /* 2.vectorized computing in the band range, the read has been reversed */
                        for (int32_t j = Bstart; j <= Bend; j += step)
                        {
                            // read nodes within the band
                            __mxxxi readChars = SIMD::load((const __mxxxi *)&readCharsInt[j - Bstart]);

                            // match or mismatch comparison
                            auto compareChar = SIMD::cmpeq(readChars, graphChar);

                            // match score vector
                            __mxxxi matchScore = SIMD::blend(compareChar, mismatch, match);

                            // avx2 register to store local maximum match score
                            __m256i max_match = matchScore;

                            // avx2 register to store local maximum delete score
                            __m256i max_delete = del;
                            // __mxxxi currentMax = SIMD::set1(INT32_MIN);

                            // match score for column j, used when only column j-1 of predecessor row is computed
                            typename SIMD::type matchJ = INT32_MIN;

                            // match & delete scores from all predecessor rows
                            for (auto k = graphLocal.offsets_out[i]; k < graphLocal.offsets_out[i + 1]; k++)
                            {
                                size_t m = graphLocal.adjcny_out[k]; // index of predecessor row
                                DPRow *prow;                         // pointer to predecessor row
                                int distance = m - i;                // distance between the current row and predecessor row
                                if (m > refEnd)
                                    continue;
                                if (distance >= MAX_PREV_ROWS)
                                    prow = fartherRows[m];
                                else
                                    prow = &dp_buffer[m % MAX_PREV_ROWS];

                                // whether the vectorized region right above has been computed
                                int pre_Bstart, pre_Bend;
                                pre_Bstart = prow->Bstart;
                                pre_Bend = prow->Bend;

                                // if the vectorized region right above has not been computed, skip this predecessor row
                                if (j < pre_Bstart || j + step - 1 > pre_Bend)
                                {
                                    // if only j-1 column is computed in predecessor row, calculate the match score for column j
                                    if (j - 1 == pre_Bend)
                                    {
                                        int matchscore = graphLocal.vertex_label[i] == read[Rstart + j] ? parameters.match : -1 * parameters.mismatch;
                                        typename SIMD::type matchj = prow->values[pre_Bend - pre_Bstart] + matchscore;
                                        if (matchj > matchJ)
                                            matchJ = matchj;
                                    }
                                    continue;
                                }

                                // calculate delete score vector
                                __m256i pre_value = _mm256_loadu_si256((__m256i *)(prow->values.data() + j - pre_Bstart));
                                __m256i del_edit = _mm256_add_epi32(pre_value, del);

                                // calculate match score vector
                                __m256i pre_valuem1;
                                if (j - 1 < pre_Bstart)
                                {
                                    __m128i lo = _mm256_castsi256_si128(pre_value);      // [a0, a1, a2, a3]
                                    __m128i hi = _mm256_extracti128_si256(pre_value, 1); // [a4, a5, a6, a7]
                                    __m128i low128 = _mm_slli_si128(lo, 4);              // [0, a0, a1, a2]
                                    __m128i part1 = _mm_srli_si128(lo, 12);              // [a3, 0, 0, 0]
                                    __m128i part2 = _mm_slli_si128(hi, 4);               // [0, a4, a5, a6]
                                    __m128i high128 = _mm_or_si128(part1, part2);        // [a3, a4, a5, a6]

                                    // combine low128 + high128 into 256-bit vector
                                    pre_valuem1 = _mm256_castsi128_si256(low128);                   // insert low half
                                    pre_valuem1 = _mm256_inserti128_si256(pre_valuem1, high128, 1); // set high half
                                }
                                else
                                {
                                    pre_valuem1 = _mm256_loadu_si256((__m256i *)(prow->values.data() + j - 1 - pre_Bstart));
                                }

                                __m256i match_edit = _mm256_add_epi32(pre_valuem1, matchScore);

                                max_delete = _mm256_max_epi32(max_delete, del_edit);
                                max_match = _mm256_max_epi32(max_match, match_edit);
                            }

                            _mm256_storeu_si256((__m256i *)(match_delete.data() + j - Bstart), _mm256_max_epi32(max_match, max_delete));
                            if (matchJ > match_delete[j - Bstart])
                                match_delete[j - Bstart] = matchJ;

                            // computer insert score and store maximum score to dp_buffer
                            for (int32_t l = j - Bstart; l < j - Bstart + step; l++)
                            {
                                if (l >= readLen - Bstart) // out of bound
                                    break;
                                int ins_edit = (l == 0) ? -parameters.ins : (current_row->values[l - 1] - parameters.ins);
                                int current_value = std::max(std::max(match_delete[l], ins_edit), 0);
                                current_row->values[l] = current_value;
                                if (current_value > current_row_max)
                                {
                                    current_row_max = current_value;
                                    current_row_max_index_left = l + Bstart;
                                }
                            }
                        }
                        current_row->max_index_left = current_row_max_index_left;
                        if (current_row_max >= bestScore)
                        {
                            bestScore = current_row_max;
                            bestCol = i;
                            bestRow = current_row_max_index_left;
                        }

                        /* 3. Drop conditions */
                        if (bestScore >= forward_bestScore)
                            break;

                        /* 4. if current row is LongHop row, save it to fatherRowsbuffer */
                        if (withLongHopLocal[i])
                        {
                            fartherRows[i]->Bstart = current_row->Bstart;
                            fartherRows[i]->Bend = current_row->Bend;
                            fartherRows[i]->max_index_left = current_row_max_index_left;
                            for (int32_t j = 0; j <= Bend - Bstart; j++)
                            {
                                fartherRows[i]->values[j] = current_row->values[j];
                            }
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
         * @brief traceback based on poa algorithm using avx2 simd instructions
         * @param R R value of each vertex of the graph
         * @param bestScoreVector
         */
        void traceback_poa_avx2(const std::vector<int32_t> &R,
                                std::vector<BestScoreInfo> &bestScoreVector)
        {
            // number of cells processed in each vector operation
            int step = SIMD::numSeqs;

            // vectorized scoring parameters
            __mxxxi match = SIMD::set1((typename SIMD::type)parameters.match);
            __mxxxi mismatch = SIMD::set1((typename SIMD::type) - 1 * parameters.mismatch);
            __mxxxi del = SIMD::set1((typename SIMD::type) - 1 * parameters.del);
            __mxxxi ins = SIMD::set1((typename SIMD::type) - 1 * parameters.ins);

            // copy graph as function variable for faster access
            const CSR_char_container graphLocal = this->graph;
            const std::vector<bool> withLongHopLocal = withLongHop;
#pragma omp parallel
            {

#pragma omp for schedule(dynamic)
                for (size_t readno = 0; readno < readSet.size(); readno++)
                {
                    const std::string &read = readSet[readno];                      // current read
                    int32_t refEnd = bestScoreVector[readno].refColumnEnd;          // ending position of reference graph
                    int32_t qryEnd = bestScoreVector[readno].qryRowEnd;             // ending position of current read
                    int32_t bestCol = bestScoreVector[readno].refColumnStart;       // starting position of reference graph
                    int32_t bestRow = qryEnd - bestScoreVector[readno].qryRowStart; // starting position of the reversed read

                    int32_t lines = refEnd - bestCol + 1;    // the number of lines between starting and ending positions
                    int Mstart, Mend;                        // maximum row range
                    int Pleft, Pright;                       // the leftmost and rightmost positions of maximum values of all predecessor rows
                    int Bstart, Bend;                        // band range in current row
                    int readLen = qryEnd + 1;                // effective read length
                    int Rvalue;                              // Rvalue of current vertex
                    int w = b + f * readLen;                 // extra bases added to both sides of the band
                    int Rstart = read.length() - qryEnd - 1; // starting index of the reversed read
                    std::string cigar;                       // base-to-base alignment result

                    // information of all lines from bestCol to refEnd
                    std::vector<DPRow> dp_band(lines);
                    for (int i = 0; i < lines; i++)
                    {
                        dp_band[i].Bstart = 0;
                        dp_band[i].Bend = 0;
                        dp_band[i].values.resize(MAX_BANDWIDTH, 0);
                    }

                    for (int32_t i = refEnd; i >= bestCol; i--)
                    {
                        /* 1. compute band range */
                        if (i == refEnd)
                            Mstart = Mend = 0;
                        else
                        {
                            Pleft = INT32_MAX;
                            Pright = INT32_MIN;
                            bool have_pre = false; // if there is any predecessor row
                            for (auto k = graphLocal.offsets_out[i]; k < graphLocal.offsets_out[i + 1]; k++)
                            {
                                size_t m = graphLocal.adjcny_out[k]; // index of predecessor row
                                DPRow *prow;                         // pointer to predecessor row
                                if (m > refEnd)
                                    continue;
                                have_pre = true;
                                prow = &dp_band[refEnd - m];

                                int pre_max_index_left = prow->max_index_left;
                                if (pre_max_index_left < Pleft)
                                    Pleft = pre_max_index_left;
                                if (pre_max_index_left > Pright)
                                    Pright = pre_max_index_left;
                            }
                            if (have_pre)
                            {
                                Mstart = Pleft + 1;
                                Mend = Pright + 1;
                            }
                            else
                                Mstart = Mend = 0;
                        }
                        Rvalue = R[i];
                        int Bstart0 = std::max(0, Mstart - w);
                        int Bend0 = std::min(readLen - 1, Mend + w);
                        Bstart = Bstart0 / step * step;
                        Bend = (Bend0 + step) / step * step - 1;
                        // the termination condition of DP
                        if (Bstart0 >= Bend0)
                            break;

                        DPRow *current_row = &dp_band[refEnd - i]; // the pointer to current row in dp_buffer

                        current_row->Bstart = Bstart;
                        current_row->Bend = Bend;
                        int current_row_max = INT32_MIN;                // maximum value in current row
                        int current_row_max_index_left;                 // index of maximum value in current row
                        std::vector<int> row_values(Bend - Bstart + 1); // store values within the band of current row

                        // current graph character
                        __mxxxi graphChar = SIMD::set1((typename SIMD::type)graphLocal.vertex_label[i]);
                        // read node corresponding to the current row
                        std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> readCharsInt(Bend - Bstart + 1);
                        for (int32_t j = Bstart; j <= Bend; j++)
                        {
                            if (j < readLen)
                                readCharsInt[j - Bstart] = read[Rstart + j];
                        }
                        if (Bend - Bstart + 1 > MAX_BANDWIDTH)
                        {
                            current_row->values.resize(Bend - Bstart + 1, 0);
                        }
                        // match & delete scores within the band
                        std::vector<int32_t, aligned_alloc<int32_t, 64>> match_delete(Bend - Bstart + 1);

                        /* 2.vectorized computing in the band range, the read has been reversed */
                        for (int32_t j = Bstart; j <= Bend; j += step)
                        {
                            // read nodes within the band
                            __mxxxi readChars = SIMD::load((const __mxxxi *)&readCharsInt[j - Bstart]);

                            // match or mismatch comparison
                            auto compareChar = SIMD::cmpeq(readChars, graphChar);

                            // match score vector
                            __mxxxi matchScore = SIMD::blend(compareChar, mismatch, match);

                            // avx2 register to store local maximum match score
                            __m256i max_match = matchScore;

                            // avx2 register to store local maximum delete score
                            __m256i max_delete = del;
                            // __mxxxi currentMax = SIMD::set1(INT32_MIN);

                            // match score for column j, used when only column j-1 of predecessor row is computed
                            typename SIMD::type matchJ = INT32_MIN;

                            // match & delete scores from all predecessor rows
                            for (auto k = graphLocal.offsets_out[i]; k < graphLocal.offsets_out[i + 1]; k++)
                            {
                                size_t m = graphLocal.adjcny_out[k]; // index of predecessor row
                                DPRow *prow;                         // pointer to predecessor row
                                int distance = m - i;                // distance between the current row and predecessor row
                                if (m > refEnd)
                                    continue;
                                prow = &dp_band[refEnd - m];

                                // whether the vectorized region right above has been computed
                                int pre_Bstart, pre_Bend;
                                pre_Bstart = prow->Bstart;
                                pre_Bend = prow->Bend;

                                // if the vectorized region right above has not been computed, skip this predecessor row
                                if (j < pre_Bstart || j + step - 1 > pre_Bend)
                                {
                                    // if only j-1 column is computed in predecessor row, calculate the match score for column j
                                    if (j - 1 == pre_Bend)
                                    {
                                        int matchscore = graphLocal.vertex_label[i] == read[Rstart + j] ? parameters.match : -1 * parameters.mismatch;
                                        typename SIMD::type matchj = prow->values[pre_Bend - pre_Bstart] + matchscore;
                                        if (matchj > matchJ)
                                            matchJ = matchj;
                                    }
                                    continue;
                                }

                                // calculate delete score vector
                                __m256i pre_value = _mm256_loadu_si256((__m256i *)(prow->values.data() + j - pre_Bstart));
                                __m256i del_edit = _mm256_add_epi32(pre_value, del);

                                // calculate match score vector
                                __m256i pre_valuem1;
                                if (j - 1 < pre_Bstart)
                                {
                                    __m128i lo = _mm256_castsi256_si128(pre_value);      // [a0, a1, a2, a3]
                                    __m128i hi = _mm256_extracti128_si256(pre_value, 1); // [a4, a5, a6, a7]
                                    __m128i low128 = _mm_slli_si128(lo, 4);              // [0, a0, a1, a2]
                                    __m128i part1 = _mm_srli_si128(lo, 12);              // [a3, 0, 0, 0]
                                    __m128i part2 = _mm_slli_si128(hi, 4);               // [0, a4, a5, a6]
                                    __m128i high128 = _mm_or_si128(part1, part2);        // [a3, a4, a5, a6]

                                    // combine low128 + high128 into 256-bit vector
                                    pre_valuem1 = _mm256_castsi128_si256(low128);                   // insert low half
                                    pre_valuem1 = _mm256_inserti128_si256(pre_valuem1, high128, 1); // set high half
                                }
                                else
                                {
                                    pre_valuem1 = _mm256_loadu_si256((__m256i *)(prow->values.data() + j - 1 - pre_Bstart));
                                }

                                __m256i match_edit = _mm256_add_epi32(pre_valuem1, matchScore);

                                max_delete = _mm256_max_epi32(max_delete, del_edit);
                                max_match = _mm256_max_epi32(max_match, match_edit);
                            }
                            _mm256_storeu_si256((__m256i *)(match_delete.data() + j - Bstart), _mm256_max_epi32(max_match, max_delete));
                            if (matchJ > match_delete[j - Bstart])
                                match_delete[j - Bstart] = matchJ;

                            // computer insert score and store maximum score to dp_buffer
                            for (int32_t l = j - Bstart; l < j - Bstart + step; l++)
                            {
                                if (l >= readLen - Bstart) // out of bound
                                    break;
                                int ins_edit = (l == 0) ? -parameters.ins : (current_row->values[l - 1] - parameters.ins);
                                int current_value = std::max(match_delete[l], ins_edit);
                                current_row->values[l] = current_value;
                                if (current_value > current_row_max)
                                {
                                    current_row_max = current_value;
                                    current_row_max_index_left = l + Bstart;
                                }
                            }
                        }
                        current_row->max_index_left = current_row_max_index_left;
                    }

                    int current_i = bestCol;
                    int current_j = bestRow;
                    while (current_i <= refEnd && current_j >= 0)
                    {
                        DPRow *current_row = &dp_band[refEnd - current_i];

                        int32_t cur_Bstart = current_row->Bstart;
                        int32_t cur_value = current_row->values[current_j - cur_Bstart];

                        if (cur_value <= 0)
                            break;

                        // compute insert score
                        int32_t ins_edit = (current_j == cur_Bstart)
                                               ? -parameters.ins
                                               : current_row->values[current_j - cur_Bstart - 1] - parameters.ins;

                        // compute match & delete scores and their positions
                        char cur_char = graph.vertex_label[current_i]; // the graph character corresponding to the current row
                        int32_t matchScore = cur_char == read[Rstart + current_j]
                                                 ? parameters.match
                                                 : -1 * parameters.mismatch;
                        int32_t maxmatch = matchScore;       // maximum match score
                        int32_t maxmatchPos = current_i;     // predecessor node corresponding to maxmatch
                        int32_t maxdelete = -parameters.del; // maximum delete score
                        int32_t maxdeletePos;                // predecessor node corresponding to maxdelete
                        for (auto k = graphLocal.offsets_out[current_i]; k < graphLocal.offsets_out[current_i + 1]; k++)
                        {
                            size_t m = graphLocal.adjcny_out[k]; // index of predecessor row
                            if (m > refEnd)
                                continue;
                            DPRow *prow = &dp_band[refEnd - m]; // pointer to predecessor row

                            int32_t pre_Bstart = prow->Bstart;
                            int32_t pre_Bend = prow->Bend;

                            // compute delete score
                            if (current_j >= pre_Bstart && current_j <= pre_Bend)
                            {
                                int32_t del_edit = prow->values[current_j - pre_Bstart] - parameters.del;
                                if (del_edit > maxdelete)
                                {
                                    maxdelete = del_edit;
                                    maxdeletePos = m;
                                }
                            }

                            // compute match score
                            if ((current_j - 1) >= pre_Bstart && (current_j - 1) <= pre_Bend)
                            {
                                int32_t match_edit = prow->values[current_j - 1 - pre_Bstart] + matchScore;
                                if (match_edit > maxmatch)
                                {
                                    maxmatch = match_edit;
                                    maxmatchPos = m;
                                }
                            }
                        }

                        if (cur_value == maxmatch)
                        {
                            if (matchScore == parameters.match)
                                cigar.push_back('=');
                            else
                                cigar.push_back('X');

                            if (maxmatchPos == current_i)
                                break;

                            current_i = maxmatchPos;
                            current_j--;
                        }
                        else if (cur_value == maxdelete)
                        {
                            cigar.push_back('D');
                            current_i = maxdeletePos;
                        }
                        else
                        {
                            assert(cur_value == ins_edit);
                            cigar.push_back('I');
                            current_j--;
                        }
                    }

                    // shorten the cigar string
                    dga::seqUtils::cigarCompact(cigar);
                    bestScoreVector[readno].cigar = cigar;
                }
            }
        }
    };

} // namespace dga
#endif