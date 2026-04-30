/**
 * @file    align_vectorized_difference.hpp
 * @brief   vectorized DP computation based on difference recurrence relations
 * @author  Qingzhe Wang <1197460504@qq.com>
 */

#ifndef GRAPH_ALIGN_VEC_DIFF_HPP
#define GRAPH_ALIGN_VEC_DIFF_HPP

#include <immintrin.h>
#include <x86intrin.h>

#include "graphLoad.hpp"
#include "csr_char.hpp"
#include "graph_iter.hpp"
#include "base_types.hpp"
#include "utils.hpp"
#include <map>
#include <vector>
using namespace std;
/* *************** */

// External includes

#define DUMMY 'B'

#if defined(HDGA_ENABLE_AVX512)
#define SIMD_REG_SIZE 512
typedef __m512i __mxxxi;
#elif defined(HDGA_ENABLE_AVX2)
#define SIMD_REG_SIZE 256
typedef __m256i __mxxxi;
#endif

namespace hdga
{
    /**
     * Parameters and SIMD instructions specific for different score precisions required
     */

    template <typename T>
    struct SimdInst
    {
    };

    template <>
    struct SimdInst<int32_t>
    {
        typedef int32_t type;

#if defined(HDGA_ENABLE_AVX512)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        typedef __mmask16 mmask_t;
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm512_add_epi32(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm512_sub_epi32(a, b); }
        static inline __mxxxi set1(int32_t a) { return _mm512_set1_epi32(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm512_set1_epi32(a); }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm512_max_epi32(a, b); }
        static inline __mxxxi zero() { return _mm512_setzero_si512(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm512_store_si512(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm512_load_si512(mem_addr); }
        static inline mmask_t cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi32_mask(a, b); }
        static inline mmask_t cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi32_mask(a, b); }
        static inline __mxxxi blend(mmask_t k, const __mxxxi &a, const __mxxxi &b) { return _mm512_mask_blend_epi32(k, a, b); }
        static inline __mxxxi mask_set1(const __mxxxi &a, mmask_t k, int32_t b) { return _mm512_mask_set1_epi32(a, k, b); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, mmask_t k, int32_t b) { return _mm512_mask_set1_epi32(a, k, b); }
        static inline mmask_t combine_mask(mmask_t k0, mmask_t k1, mmask_t k2, mmask_t k3) { return k0; }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, mmask_t k) { c0 = mask_set1_32(c0, k, val); }

#elif defined(HDGA_ENABLE_AVX2)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm256_add_epi32(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm256_sub_epi32(a, b); }
        static inline __mxxxi set1(int32_t a) { return _mm256_set1_epi32(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm256_set1_epi32(a); }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm256_max_epi32(a, b); }
        static inline __mxxxi zero() { return _mm256_setzero_si256(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm256_store_si256(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm256_load_si256(mem_addr); }
        static inline __mxxxi cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi32(a, b); }
        static inline __mxxxi cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi32(a, b); }
        static inline __mxxxi blend(const __mxxxi &k, const __mxxxi &a, const __mxxxi &b) { return _mm256_blendv_epi8(a, b, k); }
        static inline __mxxxi mask_set1(const __mxxxi &a, const __mxxxi &k, int32_t b) { return blend(k, a, set1(b)); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, const __mxxxi &k, int32_t b) { return blend(k, a, set1_32(b)); }
        static inline __mxxxi combine_mask(const __mxxxi &k0, const __mxxxi &k1, const __mxxxi &k2, const __mxxxi &k3) { return k0; }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, const __mxxxi &k) { c0 = mask_set1_32(c0, k, val); }
#endif
    };

    template <>
    struct SimdInst<int16_t>
    {
        typedef int16_t type;

#if defined(HDGA_ENABLE_AVX512)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        typedef __mmask32 mmask_t;
        typedef SimdInst<int32_t>::mmask_t mmask_T; // for 32-bit integer operations
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm512_add_epi16(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm512_sub_epi16(a, b); }
        static inline __mxxxi set1(int16_t a) { return _mm512_set1_epi16(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm512_set1_epi32(a); }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm512_max_epi16(a, b); }
        static inline __mxxxi zero() { return _mm512_setzero_si512(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm512_store_si512(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm512_load_si512(mem_addr); }
        static inline mmask_t cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi16_mask(a, b); }
        static inline mmask_T cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi32_mask(a, b); }
        static inline __mxxxi blend(mmask_t k, const __mxxxi &a, const __mxxxi &b) { return _mm512_mask_blend_epi16(k, a, b); }
        static inline __mxxxi mask_set1(const __mxxxi &a, mmask_t k, int16_t b) { return _mm512_mask_set1_epi16(a, k, b); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, mmask_T k, int32_t b) { return _mm512_mask_set1_epi32(a, k, b); }
        static inline mmask_t combine_mask(mmask_T k0, mmask_T k1, mmask_T k2, mmask_T k3)
        {
            return ((mmask_t)k0) | ((mmask_t)k1) << 16;
        }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, mmask_t k)
        {
            c0 = mask_set1_32(c0, k, val);
            c1 = mask_set1_32(c1, k >> 16, val);
        }

#elif defined(HDGA_ENABLE_AVX2)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm256_add_epi16(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm256_sub_epi16(a, b); }
        static inline __mxxxi set1(int16_t a) { return _mm256_set1_epi16(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm256_set1_epi32(a); }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm256_max_epi16(a, b); }
        static inline __mxxxi zero() { return _mm256_setzero_si256(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm256_store_si256(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm256_load_si256(mem_addr); }
        static inline __mxxxi cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi16(a, b); }
        static inline __mxxxi cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi32(a, b); }
        static inline __mxxxi blend(const __mxxxi &k, const __mxxxi &a, const __mxxxi &b) { return _mm256_blendv_epi8(a, b, k); }
        static inline __mxxxi mask_set1(const __mxxxi &a, const __mxxxi &k, int16_t b) { return blend(k, a, set1(b)); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, const __mxxxi &k, int32_t b) { return blend(k, a, set1_32(b)); }
        static inline __mxxxi combine_mask(const __mxxxi &k0, const __mxxxi &k1, const __mxxxi &k2, const __mxxxi &k3)
        {
            return _mm256_permute4x64_epi64(_mm256_packs_epi32(k0, k1), 0xd8);
        }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, const __mxxxi &k)
        {
            c0 = mask_set1_32(c0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(k)), val);
            c1 = mask_set1_32(c1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(k, 1)), val);
        }
#endif
    };

    template <>
    struct SimdInst<int8_t>
    {
        typedef int8_t type;

#if defined(HDGA_ENABLE_AVX512)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        typedef __mmask64 mmask_t;
        typedef SimdInst<int32_t>::mmask_t mmask_T; // for 32-bit integer operations
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm512_add_epi8(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm512_sub_epi8(a, b); }
        static inline __mxxxi set1(int8_t a) { return _mm512_set1_epi8(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm512_set1_epi32(a); }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm512_max_epi8(a, b); }
        static inline __mxxxi zero() { return _mm512_setzero_si512(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm512_store_si512(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm512_load_si512(mem_addr); }
        static inline mmask_t cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi8_mask(a, b); }
        static inline mmask_T cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi32_mask(a, b); }
        static inline __mxxxi blend(mmask_t k, const __mxxxi &a, const __mxxxi &b) { return _mm512_mask_blend_epi8(k, a, b); }
        static inline __mxxxi mask_set1(const __mxxxi &a, mmask_t k, int8_t b) { return _mm512_mask_set1_epi8(a, k, b); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, mmask_T k, int32_t b) { return _mm512_mask_set1_epi32(a, k, b); }
        static inline mmask_t combine_mask(mmask_T k0, mmask_T k1, mmask_T k2, mmask_T k3)
        {
            return ((mmask_t)k0) | ((mmask_t)k1) << 16 | ((mmask_t)k2) << 32 | ((mmask_t)k3) << 48;
        }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, mmask_t k)
        {
            c0 = mask_set1_32(c0, k, val);
            c1 = mask_set1_32(c1, k >> 16, val);
            c2 = mask_set1_32(c2, k >> 32, val);
            c3 = mask_set1_32(c3, k >> 48, val);
        }

#elif defined(HDGA_ENABLE_AVX2)
        static constexpr int numSeqs = SIMD_REG_SIZE / (8 * sizeof(type));
        static inline __mxxxi add(const __mxxxi &a, const __mxxxi &b) { return _mm256_add_epi8(a, b); }
        static inline __mxxxi add_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_add_epi32(a, b); }
        static inline __mxxxi sub(const __mxxxi &a, const __mxxxi &b) { return _mm256_sub_epi8(a, b); }
        static inline __mxxxi sub_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_sub_epi32(a, b); }
        static inline __mxxxi set1(int8_t a) { return _mm256_set1_epi8(a); }
        static inline __mxxxi set1_32(int32_t a) { return _mm256_set1_epi32(a); }
        static inline __mxxxi setr_32(int8_t *a)
        {
            return _mm256_setr_epi32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        }
        static inline __mxxxi max(const __mxxxi &a, const __mxxxi &b) { return _mm256_max_epi8(a, b); }
        static inline __mxxxi max_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_max_epi32(a, b); }
        static inline __mxxxi zero() { return _mm256_setzero_si256(); }
        static inline void store(__mxxxi *mem_addr, const __mxxxi &a) { _mm256_store_si256(mem_addr, a); }
        static inline __mxxxi load(const __mxxxi *mem_addr) { return _mm256_load_si256(mem_addr); }
        static inline __mxxxi cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi8(a, b); }
        static inline __mxxxi cmpeq_32(const __mxxxi &a, const __mxxxi &b) { return _mm256_cmpeq_epi32(a, b); }
        static inline __mxxxi blend(const __mxxxi &k, const __mxxxi &a, const __mxxxi &b) { return _mm256_blendv_epi8(a, b, k); }
        static inline __mxxxi mask_set1(const __mxxxi &a, const __mxxxi &k, int8_t b) { return blend(k, a, set1(b)); }
        static inline __mxxxi mask_set1_32(const __mxxxi &a, const __mxxxi &k, int32_t b) { return blend(k, a, set1_32(b)); }
        // 将4个8x32bit向量寄存器，每个通道截取低8位，合并成32x8bit寄存器
        static inline __mxxxi combine_reg(const __mxxxi &a, const __mxxxi &b, const __mxxxi &c, const __mxxxi &d)
        {
            __mxxxi permuteOrder = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            return _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d)), permuteOrder);
        }
        static inline __mxxxi combine_mask(const __mxxxi &k0, const __mxxxi &k1, const __mxxxi &k2, const __mxxxi &k3)
        {
            __mxxxi permuteOrder = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            return _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(k0, k1), _mm256_packs_epi32(k2, k3)), permuteOrder);
        }
        static inline void update_cols(__mxxxi &c0, __mxxxi &c1, __mxxxi &c2, __mxxxi &c3, int32_t val, const __mxxxi &k)
        {
            c0 = mask_set1_32(c0, _mm256_cvtepi8_epi32(_mm256_castsi256_si128(k)), val);
            c1 = mask_set1_32(c1, _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(k, 1))), val);
            c2 = mask_set1_32(c2, _mm256_cvtepi8_epi32(_mm256_extracti128_si256(k, 1)), val);
            c3 = mask_set1_32(c3, _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(k, 3))), val);
        }
#endif
    };

    /**
     * @brief   Supports phase 1 DP in forward direction
     */
    template <typename SIMD>
    class Phase1_Vectorized_Difference
    {
    private:
        // reference graph
        const CSR_char_container &graph;

        // pre-compute which graph vertices are connected with hop longer
        // than 'blockWidth'
        std::vector<bool> withLongHop;

        // for converting input reads into SOA to enable vectorization
        std::vector<char> readSetSOA;

        // cumulative read batch sizes
        std::vector<size_t> readSetSOAPrefixSum;

        // input reads
        const std::vector<std::string> &readSet;

        // read lengths in their 'sorted order'
        std::vector<size_t> sortedReadLengths;

        // sorted permutation order of input reads
        std::vector<size_t> sortedReadOrder;

        // input parameters (e.g., scoring scheme)
        const Parameters &parameters;

    public:
        // small temporary storage buffer for DP scores
        // should be a power of 2
        static constexpr size_t blockWidth = 8;

        // process these many vertical cells in a go
        // should be a power of 2
        static constexpr size_t blockHeight = 16;

        /**
         * @brief                   public constructor
         * @param[in]   readSet     vector of input query sequences to align
         * @param[in]   g           input reference graph
         */
        Phase1_Vectorized_Difference(const std::vector<std::string> &readSet,
                                     const CSR_char_container &g,
                                     const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->sortReadsForLoadBalance();
            this->convertToSOA();
            this->computeLongHops();
        };

        /**
         * @brief                                 wrapper function for phase 1 routine
         * @param[out]  outputBestScoreVector     vector to keep value and location of best scores,
         *                                        vector size is same as count of the reads
         * @note                                  this class won't care about rev. complement of sequences
         */
        template <typename Vec>
        void alignToDAGLocal_Phase1_vectorized_difference_wrapper(Vec &outputBestScoreVector) const
        {
            assert(outputBestScoreVector.size() == readSet.size());

            std::size_t countReadBatches = std::ceil(readSet.size() * 1.0 / SIMD::numSeqs);

            // column location value requires int32_t type, so may need >1 register per read batch
            constexpr size_t colValuesPerRegister = SIMD_REG_SIZE / (8 * sizeof(int32_t));
            constexpr size_t colRegistersCountPerBatch = SIMD::numSeqs / colValuesPerRegister;

            static_assert(colRegistersCountPerBatch == 1 ||
                              colRegistersCountPerBatch == 2 ||
                              colRegistersCountPerBatch == 4,
                          "has to be either 1, 2 or 4");

            // modified containers to hold best-score info
            // vector elements aligned to 64 byte boundaries
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>
                _bestScoreVector(countReadBatches * colRegistersCountPerBatch);
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>
                _bestScoreColVector(countReadBatches * colRegistersCountPerBatch);
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>
                _bestScoreRowVector(countReadBatches * colRegistersCountPerBatch);

            // execute the alignment routine
            this->alignToDAGLocal_Phase1_vectorized_difference(_bestScoreVector, _bestScoreColVector, _bestScoreRowVector);

            // when debugging for low-precision
#ifdef DEBUG
            for (auto &e : _bestScoreVector)
                simdUtils<typename SIMD::type>::print_avx_num(e);

            for (auto &e : _bestScoreRowVector)
                simdUtils<typename SIMD::type>::print_avx_num(e);

            for (auto &e : _bestScoreColVector)
                simdUtils<int32_t>::print_avx_num(e);
#endif

            // parse best scores from vector registers
            std::vector<int32_t, aligned_alloc<int32_t, 64>> storeScores(SIMD::numSeqs);
            std::vector<int32_t, aligned_alloc<int32_t, 64>> storeRows(SIMD::numSeqs);
            std::vector<int32_t, aligned_alloc<int32_t, 64>> storeCols(SIMD::numSeqs);

            for (size_t i = 0; i < countReadBatches; i++)
            {
                for (size_t j = 0; j < colRegistersCountPerBatch; j++)
                {
                    SIMD::store((__mxxxi *)&storeScores[j * colValuesPerRegister],
                                _bestScoreVector[i * colRegistersCountPerBatch + j]);
                    SIMD::store((__mxxxi *)&storeRows[j * colValuesPerRegister],
                                _bestScoreRowVector[i * colRegistersCountPerBatch + j]);
                    SIMD::store((__mxxxi *)&storeCols[j * colValuesPerRegister],
                                _bestScoreColVector[i * colRegistersCountPerBatch + j]);
                }

                for (size_t j = 0; j < SIMD::numSeqs; j++)
                {
                    if (i * SIMD::numSeqs + j < readSet.size())
                    {
                        auto originalReadId = sortedReadOrder[i * SIMD::numSeqs + j];

                        outputBestScoreVector[originalReadId].score = storeScores[j];
                        outputBestScoreVector[originalReadId].refColumnEnd = storeCols[j];
                        outputBestScoreVector[originalReadId].qryRowEnd = storeRows[j];

#ifdef DEBUG
                        std::cout << "INFO, hdga::Phase1_Vectorized::alignToDAGLocal_Phase1_vectorized_wrapper, read # " << originalReadId << ",  score = " << (int)storeScores[j] << ", qryRowEnd = " << (int)storeRows[j] << ", refColumnEnd = " << (int)storeCols[j] << "\n";
#endif
                    }
                }
            }
        }

    private:
    private:
    private:
        /**
         * @brief       compute the sorted order of sequences for load balancing
         * @details     sorting is done in decreasing length order
         */
        void sortReadsForLoadBalance()
        {
            typedef std::pair<size_t, size_t> pair_t;

            // vector of tuples of length and index of reads
            std::vector<pair_t> lengthTuples;

            for (size_t i = 0; i < this->readSet.size(); i++)
                lengthTuples.emplace_back(readSet[i].length(), i);

            // sort in descending order, longer reads first
            std::sort(lengthTuples.begin(), lengthTuples.end(), [](const pair_t &left, const pair_t &right)
                      { return left.first > right.first || (left.first == right.first && left.second < right.second); });

            for (auto &e : lengthTuples)
            {
                this->sortedReadLengths.push_back(e.first);
                this->sortedReadOrder.push_back(e.second);
            }
        }

        /**
         * @brief       convert input reads characters into SOA to enable vectorization
         */
        void convertToSOA()
        {
            assert(readSet.size() > 0);
            assert(sortedReadOrder.size() == readSet.size());
            assert(readSetSOA.size() == 0);

            /**
             * Requirements from the padding process
             * - each read length be a multiple of 'blockHeight'
             * - count of reads be a multiple of SIMD::numSeqs
             * - each batch of SIMD::numSeqs reads should be of equal length
             */

            // re-arrange read characters for vectorized processing;
            // SIMD::numSeqs reads at a time, sorted in decreasing order by length

            auto readCount = readSet.size();

            readSetSOAPrefixSum.push_back(0);

            for (size_t i = 0; i < readCount; i += SIMD::numSeqs)
            {
                auto batchLength = readSet[sortedReadOrder[i]].length();          // longest read in this batch
                batchLength += blockHeight - 1 - (batchLength - 1) % blockHeight; // round-up

                for (size_t j = 0; j < batchLength; j++)
                    for (size_t k = 0; k < SIMD::numSeqs; k++)
                        if (i + k < readCount && j < readSet[sortedReadOrder[i + k]].length())
                            readSetSOA.push_back(readSet[sortedReadOrder[i + k]][j]);
                        else
                            readSetSOA.push_back(DUMMY);

                readSetSOAPrefixSum.push_back(readSetSOA.size());
            }

            std::size_t countReadBatches = std::ceil(readSet.size() * 1.0 / SIMD::numSeqs);
            assert(readSetSOAPrefixSum.size() == countReadBatches + 1);
        }

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
                    if (to_pos - from_pos >= this->blockWidth)
                        this->withLongHop[from_pos] = true;
                }
            }

#ifdef DEBUG
            auto trueCount = std::count(withLongHop.begin(), withLongHop.end(), true);
            std::cout << "INFO, hdga::Phase1_Vectorized::computeLongHops, fraction of hops that are long: " << trueCount * 1.0 / graph.numVertices << "\n";
#endif
        }

        /**
         * @brief                         execute first phase of alignment i.e. compute DP and
         *                                find locations of the best alignment of each read
         * @param[out]  bestScores        best DP scores of reads
         * @param[out]  bestCols          columns where best alignment ends (for traceback later)
         * @param[out]  bestRows          rows where best alignment ends
         */
        template <typename Vec>
        void alignToDAGLocal_Phase1_vectorized_difference(Vec &bestScores, Vec &bestCols, Vec &bestRows) const
        {
            // the number of reads
            std::size_t readCount = readSet.size();
            // BatchSize
            std::size_t countReadBatches = std::ceil(readCount * 1.0 / SIMD::numSeqs);

            // the number of int32_t that could be stored in each register
            constexpr size_t colValuesPerRegister = SIMD_REG_SIZE / (8 * sizeof(int32_t));
            // the number of colregister required in each batch
            constexpr size_t colRegistersCountPerBatch = SIMD::numSeqs / colValuesPerRegister;

            assert(bestScores.size() == countReadBatches * colRegistersCountPerBatch);
            assert(bestCols.size() == countReadBatches * colRegistersCountPerBatch);
            assert(bestRows.size() == countReadBatches * colRegistersCountPerBatch);

#ifdef VTUNE_SUPPORT
            __itt_resume();
#endif

            // initialize bestScores, bestCols, bestRows, filled by zero
            std::fill(bestScores.begin(), bestScores.end(), SIMD::zero());
            std::fill(bestCols.begin(), bestCols.end(), SIMD::zero());
            std::fill(bestRows.begin(), bestRows.end(), SIMD::zero());

            // initialize vectors of penalty scores
            __mxxxi match512 = SIMD::set1((typename SIMD::type)parameters.match);
            __mxxxi mismatch512 = SIMD::set1((typename SIMD::type) - 1 * parameters.mismatch);
            __mxxxi del512 = SIMD::set1((typename SIMD::type) - 1 * parameters.del);
            __mxxxi del512_32 = SIMD::set1_32((int32_t)-1 * parameters.del);
            __mxxxi ins512 = SIMD::set1((typename SIMD::type) - 1 * parameters.ins);
            __mxxxi ins512_32 = SIMD::set1_32((int32_t)-1 * parameters.ins);

            // time usage in each thread
            std::vector<double> threadTimings(omp_get_max_threads(), 0);
            // lengths of reads processed by each thread
            std::vector<double> threadReadLengths(omp_get_max_threads(), 0);

            // copy graph as function variable for faster access
            const CSR_char_container graphLocal = this->graph;
            const std::vector<bool> withLongHopLocal = withLongHop;

            // count predecessors of each vertex in the graph
            vector<int> OriginalIndex(graphLocal.numVertices); // the index of the first predecessor of each vertex
            int index = 0;                                     // total number of predecessors in the graph
            int maxInDegree = 0;                               // maximum indegree of all vertices
            for (int32_t k = 0; k < graphLocal.numVertices; k++)
            {
                if (graphLocal.offsets_in[k] < graphLocal.offsets_in[k + 1])
                {
                    int num_Indegree = graphLocal.offsets_in[k + 1] - graphLocal.offsets_in[k]; // the indegree of current vertex
                    OriginalIndex[k] = index;
                    index += num_Indegree;
                    maxInDegree = max(maxInDegree, num_Indegree);
                }
            }

#pragma omp parallel
            {
#pragma omp barrier
                // the start time of each thread
                threadTimings[omp_get_thread_num()] = omp_get_wtime();

                using AlignedVecType = std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>;

                // number of LongHop nodes in the local graph
                std::size_t countLongHops = std::count(withLongHopLocal.begin(), withLongHopLocal.end(), true);
                // store for LongHop nodes in H matrix
                AlignedVecType fartherColumnsBuffer_0(countLongHops * this->blockHeight);
                AlignedVecType fartherColumnsBuffer_1(countLongHops * this->blockHeight);
                AlignedVecType fartherColumnsBuffer_2(countLongHops * this->blockHeight);
                AlignedVecType fartherColumnsBuffer_3(countLongHops * this->blockHeight);
                // store for LongHop nodes in U matrix
                AlignedVecType UfartherColumnsBuffer(countLongHops * this->blockHeight);

                // store pointers to fartherColumnsBuffer
                std::vector<__mxxxi *> fartherColumns_0(graphLocal.numVertices);
                std::vector<__mxxxi *> fartherColumns_1(graphLocal.numVertices);
                std::vector<__mxxxi *> fartherColumns_2(graphLocal.numVertices);
                std::vector<__mxxxi *> fartherColumns_3(graphLocal.numVertices);
                // store pointers to UfartherColumnsBuffer
                std::vector<__mxxxi *> UfartherColumns(graphLocal.numVertices);
                {
                    size_t j = 0;

                    for (int32_t i = 0; i < graphLocal.numVertices; i++)
                        if (withLongHopLocal[i])
                        {
                            fartherColumns_0[i] = &fartherColumnsBuffer_0[j * this->blockHeight];
                            fartherColumns_1[i] = &fartherColumnsBuffer_1[j * this->blockHeight];
                            fartherColumns_2[i] = &fartherColumnsBuffer_2[j * this->blockHeight];
                            fartherColumns_3[i] = &fartherColumnsBuffer_3[j * this->blockHeight];
                            UfartherColumns[i] = &UfartherColumnsBuffer[j * this->blockHeight];
                            j++;
                        }
                }

                // store for nearby columns in H matrix
                AlignedVecType nearbyColumnsBuffer_0(this->blockWidth * this->blockHeight);
                AlignedVecType nearbyColumnsBuffer_1(this->blockWidth * this->blockHeight);
                AlignedVecType nearbyColumnsBuffer_2(this->blockWidth * this->blockHeight);
                AlignedVecType nearbyColumnsBuffer_3(this->blockWidth * this->blockHeight);
                // store for nearby columns in U matrix
                AlignedVecType UnearbyColumnsBuffer(this->blockWidth * this->blockHeight);
                // store for V matrix
                AlignedVecType V0(maxInDegree);
                AlignedVecType V1(maxInDegree);
                int lastV = 1;

                // store pointers to nearbyColumnsBuffer
                std::vector<__mxxxi *> nearbyColumns_0(this->blockWidth);
                std::vector<__mxxxi *> nearbyColumns_1(this->blockWidth);
                std::vector<__mxxxi *> nearbyColumns_2(this->blockWidth);
                std::vector<__mxxxi *> nearbyColumns_3(this->blockWidth);
                // store pointers to UnearbyColumnsBuffer
                std::vector<__mxxxi *> UnearbyColumns(this->blockWidth);
                {
                    for (std::size_t i = 0; i < this->blockWidth; i++)
                    {
                        nearbyColumns_0[i] = &nearbyColumnsBuffer_0[i * this->blockHeight];
                        nearbyColumns_1[i] = &nearbyColumnsBuffer_1[i * this->blockHeight];
                        nearbyColumns_2[i] = &nearbyColumnsBuffer_2[i * this->blockHeight];
                        nearbyColumns_3[i] = &nearbyColumnsBuffer_3[i * this->blockHeight];
                        UnearbyColumns[i] = &UnearbyColumnsBuffer[i * this->blockHeight];
                    }
                }

                // store for last row in previous batch
                AlignedVecType lastBatchRowBuffer_0(2 * graphLocal.numVertices);
                AlignedVecType lastBatchRowBuffer_1(2 * graphLocal.numVertices);
                AlignedVecType lastBatchRowBuffer_2(2 * graphLocal.numVertices);
                AlignedVecType lastBatchRowBuffer_3(2 * graphLocal.numVertices);
                AlignedVecType VlastBatchRowBuffer(2 * graphLocal.numEdges);

                // pointer to last row in previous batch
                std::vector<__mxxxi *> lastBatchRow_0(2);
                std::vector<__mxxxi *> lastBatchRow_1(2);
                std::vector<__mxxxi *> lastBatchRow_2(2);
                std::vector<__mxxxi *> lastBatchRow_3(2);
                // pointer to last row in previous batch for V matrix
                std::vector<__mxxxi *> VlastBatchRow(2);
                {
                    lastBatchRow_0[0] = &lastBatchRowBuffer_0[0];
                    lastBatchRow_0[1] = &lastBatchRowBuffer_0[graphLocal.numVertices];
                    lastBatchRow_1[0] = &lastBatchRowBuffer_1[0];
                    lastBatchRow_1[1] = &lastBatchRowBuffer_1[graphLocal.numVertices];
                    lastBatchRow_2[0] = &lastBatchRowBuffer_2[0];
                    lastBatchRow_2[1] = &lastBatchRowBuffer_2[graphLocal.numVertices];
                    lastBatchRow_3[0] = &lastBatchRowBuffer_3[0];
                    lastBatchRow_3[1] = &lastBatchRowBuffer_3[graphLocal.numVertices];
                    VlastBatchRow[0] = &VlastBatchRowBuffer[0];
                    VlastBatchRow[1] = &VlastBatchRowBuffer[graphLocal.numEdges];
                }

                // readChars for current block
                std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> readCharsInt(SIMD::numSeqs * this->blockHeight);

#pragma omp for schedule(dynamic) nowait
                for (size_t i = 0; i < countReadBatches; i++)
                {
                    // initialize best score and location for the current read batch
                    __mxxxi bestScores512_0 = SIMD::zero();
                    __mxxxi bestScores512_1 = SIMD::zero();
                    __mxxxi bestScores512_2 = SIMD::zero();
                    __mxxxi bestScores512_3 = SIMD::zero();
                    __mxxxi bestRows512_0 = SIMD::zero();
                    __mxxxi bestRows512_1 = SIMD::zero();
                    __mxxxi bestRows512_2 = SIMD::zero();
                    __mxxxi bestRows512_3 = SIMD::zero();
                    __mxxxi bestCols512_0 = SIMD::zero();
                    __mxxxi bestCols512_1 = SIMD::zero();
                    __mxxxi bestCols512_2 = SIMD::zero();
                    __mxxxi bestCols512_3 = SIMD::zero();

                    // the longest read length in the current batch
                    int32_t qryBatchLength = readSet[sortedReadOrder[i * SIMD::numSeqs]].length();
                    // round up to multiple of blockHeight
                    qryBatchLength += this->blockHeight - 1 - (qryBatchLength - 1) % this->blockHeight;

                    threadReadLengths[omp_get_thread_num()] += qryBatchLength * 256 / 8;

                    for (int32_t j = 0; j < qryBatchLength; j += this->blockHeight)
                    {
                        size_t loopJ = j / (this->blockHeight);

                        // load read characters for the j-th block for the current read batch
                        for (int32_t k = 0; k < SIMD::numSeqs * this->blockHeight; k++)
                        {
                            readCharsInt[k] = readSetSOA[readSetSOAPrefixSum[i] + j * SIMD::numSeqs + k];
                        }

                        for (int32_t k = 0; k < graphLocal.numVertices; k++)
                        {
                            // current graphchar(8bit)
                            __mxxxi graphChar = SIMD::set1((typename SIMD::type)graphLocal.vertex_label[k]);

                            __mxxxi currentMax512_0;
                            __mxxxi currentMax512_1;
                            __mxxxi currentMax512_2;
                            __mxxxi currentMax512_3;
                            __mxxxi currentU, currentV;

                            for (size_t l = 0; l < this->blockHeight; l++)
                            {
                                // load read characters for the l-th row in the current block
                                __mxxxi readChars = SIMD::load((const __mxxxi *)&readCharsInt[l * SIMD::numSeqs]);

                                // compare if readChars == graphChar
                                auto compareChar = SIMD::cmpeq(readChars, graphChar);

                                // Δij
                                __mxxxi sub512 = SIMD::blend(compareChar, mismatch512, match512);

                                // iterate predecessors of vertex k
                                // case1: the first row of the first block (the first row of the entire matrix), only compute V matrix
                                if (l == 0 && loopJ == 0)
                                {
                                    currentMax512_0 = SIMD::zero();
                                    currentMax512_1 = SIMD::zero();
                                    currentMax512_2 = SIMD::zero();
                                    currentMax512_3 = SIMD::zero();

                                    // divide sub512 into four 32-bit vectors
                                    __mxxxi sub512_H0, sub512_H1, sub512_H2, sub512_H3;
                                    sub512_H0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(sub512));
                                    sub512_H1 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 1)));
                                    sub512_H2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(sub512, 1));
                                    sub512_H3 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 3)));

                                    // update currentMax512
                                    currentMax512_0 = SIMD::max_32(currentMax512_0, sub512_H0);
                                    currentMax512_1 = SIMD::max_32(currentMax512_1, sub512_H1);
                                    currentMax512_2 = SIMD::max_32(currentMax512_2, sub512_H2);
                                    currentMax512_3 = SIMD::max_32(currentMax512_3, sub512_H3);

                                    // calculate Hi,j
                                    for (size_t m = graphLocal.offsets_in[k]; m < graphLocal.offsets_in[k + 1]; m++)
                                    {
                                        // Hi,k - Δins
                                        __mxxxi insEdit_0;
                                        __mxxxi insEdit_1;
                                        __mxxxi insEdit_2;
                                        __mxxxi insEdit_3;
                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                        {
                                            insEdit_0 = SIMD::add_32(nearbyColumns_0[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512_32);
                                            insEdit_1 = SIMD::add_32(nearbyColumns_1[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512_32);
                                            insEdit_2 = SIMD::add_32(nearbyColumns_2[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512_32);
                                            insEdit_3 = SIMD::add_32(nearbyColumns_3[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512_32);
                                        }
                                        else
                                        {
                                            insEdit_0 = SIMD::add_32(fartherColumns_0[graphLocal.adjcny_in[m]][l], ins512_32);
                                            insEdit_1 = SIMD::add_32(fartherColumns_1[graphLocal.adjcny_in[m]][l], ins512_32);
                                            insEdit_2 = SIMD::add_32(fartherColumns_2[graphLocal.adjcny_in[m]][l], ins512_32);
                                            insEdit_3 = SIMD::add_32(fartherColumns_3[graphLocal.adjcny_in[m]][l], ins512_32);
                                        }

                                        // update currentMax512
                                        currentMax512_0 = SIMD::max_32(currentMax512_0, insEdit_0);
                                        currentMax512_1 = SIMD::max_32(currentMax512_1, insEdit_1);
                                        currentMax512_2 = SIMD::max_32(currentMax512_2, insEdit_2);
                                        currentMax512_3 = SIMD::max_32(currentMax512_3, insEdit_3);
                                    }

                                    // calculate V(k)i,j
                                    for (size_t m = graphLocal.offsets_in[k], i = 0; m < graphLocal.offsets_in[k + 1]; m++, i++)
                                    {
                                        // V(k)i,j = Hi,j - Hi,k
                                        __mxxxi currentV32_0;
                                        __mxxxi currentV32_1;
                                        __mxxxi currentV32_2;
                                        __mxxxi currentV32_3;
                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                        {
                                            currentV32_0 = SIMD::sub_32(currentMax512_0, nearbyColumns_0[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                            currentV32_1 = SIMD::sub_32(currentMax512_1, nearbyColumns_1[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                            currentV32_2 = SIMD::sub_32(currentMax512_2, nearbyColumns_2[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                            currentV32_3 = SIMD::sub_32(currentMax512_3, nearbyColumns_3[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                        }
                                        else
                                        {
                                            currentV32_0 = SIMD::sub_32(currentMax512_0, fartherColumns_0[graphLocal.adjcny_in[m]][l]);
                                            currentV32_1 = SIMD::sub_32(currentMax512_1, fartherColumns_1[graphLocal.adjcny_in[m]][l]);
                                            currentV32_2 = SIMD::sub_32(currentMax512_2, fartherColumns_2[graphLocal.adjcny_in[m]][l]);
                                            currentV32_3 = SIMD::sub_32(currentMax512_3, fartherColumns_3[graphLocal.adjcny_in[m]][l]);
                                        }

                                        // integrate four 32-bit V into one 8-bit V
                                        currentV = SIMD::combine_reg(currentV32_0, currentV32_1, currentV32_2, currentV32_3);

                                        // save currentV to V0
                                        V0[i] = currentV;
                                    }
                                    lastV = 0;
                                }

                                // case2: first rows of other blocks, need last row of previous block
                                else if (l == 0)
                                {
                                    // calculate Hi,j and ui,j directly when there is no predecessor node
                                    if (graphLocal.offsets_in[k] >= graphLocal.offsets_in[k + 1])
                                    {
                                        currentMax512_0 = SIMD::zero();
                                        currentMax512_1 = SIMD::zero();
                                        currentMax512_2 = SIMD::zero();
                                        currentMax512_3 = SIMD::zero();

                                        // divide Δij
                                        __mxxxi sub512_H0, sub512_H1, sub512_H2, sub512_H3;
                                        sub512_H0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(sub512));
                                        sub512_H1 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 1)));
                                        sub512_H2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(sub512, 1));
                                        sub512_H3 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 3)));

                                        // update currentMax512
                                        currentMax512_0 = SIMD::max_32(currentMax512_0, sub512_H0);
                                        currentMax512_1 = SIMD::max_32(currentMax512_1, sub512_H1);
                                        currentMax512_2 = SIMD::max_32(currentMax512_2, sub512_H2);
                                        currentMax512_3 = SIMD::max_32(currentMax512_3, sub512_H3);

                                        // Hi-1,j - Δdel
                                        __mxxxi delEdit_0 = SIMD::add_32(lastBatchRow_0[(loopJ - 1) & 1][k], del512_32);
                                        __mxxxi delEdit_1 = SIMD::add_32(lastBatchRow_1[(loopJ - 1) & 1][k], del512_32);
                                        __mxxxi delEdit_2 = SIMD::add_32(lastBatchRow_2[(loopJ - 1) & 1][k], del512_32);
                                        __mxxxi delEdit_3 = SIMD::add_32(lastBatchRow_3[(loopJ - 1) & 1][k], del512_32);

                                        // update currentMax512
                                        currentMax512_0 = SIMD::max_32(currentMax512_0, delEdit_0);
                                        currentMax512_1 = SIMD::max_32(currentMax512_1, delEdit_1);
                                        currentMax512_2 = SIMD::max_32(currentMax512_2, delEdit_2);
                                        currentMax512_3 = SIMD::max_32(currentMax512_3, delEdit_3);

                                        // ui,j = Hi,j - Hi-1,j
                                        // integrate four 32-bit U into one 8-bit U
                                        __mxxxi currentU32_0 = SIMD::sub_32(currentMax512_0, lastBatchRow_0[(loopJ - 1) & 1][k]);
                                        __mxxxi currentU32_1 = SIMD::sub_32(currentMax512_1, lastBatchRow_1[(loopJ - 1) & 1][k]);
                                        __mxxxi currentU32_2 = SIMD::sub_32(currentMax512_2, lastBatchRow_2[(loopJ - 1) & 1][k]);
                                        __mxxxi currentU32_3 = SIMD::sub_32(currentMax512_3, lastBatchRow_3[(loopJ - 1) & 1][k]);
                                        currentU = SIMD::combine_reg(currentU32_0, currentU32_1, currentU32_2, currentU32_3);

                                        UnearbyColumns[k & (blockWidth - 1)][l] = currentU;
                                        // save currentU to UnearbyColumns and UfartherColumns if longHop
                                        if (withLongHopLocal[k])
                                            UfartherColumns[k][l] = currentU;
                                    }

                                    bool uCalculated = false;

                                    // iterate predecessors of vertex k
                                    for (size_t m = graphLocal.offsets_in[k], i = 0; m < graphLocal.offsets_in[k + 1]; m++, i++)
                                    {
                                        // initialize currentZ, delEdit = V(k)i-1,j - Δdel
                                        __mxxxi currentZ = SIMD::add(VlastBatchRow[(loopJ - 1) & 1][OriginalIndex[k] + m - graphLocal.offsets_in[k]], del512);

                                        // iterate predecessors mm of vertex k
                                        for (size_t mm = graphLocal.offsets_in[k]; mm < graphLocal.offsets_in[k + 1]; mm++)
                                        {
                                            // when k = k'
                                            if (mm == m)
                                            {
                                                // subStEdit = Δij
                                                __mxxxi substEdit = sub512;

                                                // insEdit = Ui,k' - Δins = Ui,k - Δins
                                                __mxxxi insEdit;
                                                if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                                    insEdit = SIMD::add(UnearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512);
                                                else
                                                    insEdit = SIMD::add(UfartherColumns[graphLocal.adjcny_in[m]][l], ins512);

                                                // update maximum of Z(k)
                                                currentZ = SIMD::max(currentZ, substEdit);
                                                currentZ = SIMD::max(currentZ, insEdit);
                                            }
                                            else
                                            {
                                                // V(k)i-1,j - V(k')i-1,j
                                                __mxxxi VsubEdit = SIMD::sub(VlastBatchRow[(loopJ - 1) & 1][OriginalIndex[k] + m - graphLocal.offsets_in[k]], VlastBatchRow[(loopJ - 1) & 1][OriginalIndex[k] + mm - graphLocal.offsets_in[k]]);

                                                // subStEdit = V(k)i-1,j - V(k')i-1,j + Δij
                                                __mxxxi substEdit = SIMD::add(VsubEdit, sub512);

                                                // insEdit = Ui,k' + V(k)i-1,j - V(k')i-1,j - Δins
                                                __mxxxi insEdit;
                                                if (k - graphLocal.adjcny_in[mm] < this->blockWidth)
                                                    insEdit = SIMD::add(UnearbyColumns[graphLocal.adjcny_in[mm] & (blockWidth - 1)][l], VsubEdit);
                                                else
                                                    insEdit = SIMD::add(UfartherColumns[graphLocal.adjcny_in[mm]][l], VsubEdit);
                                                insEdit = SIMD::add(insEdit, ins512);

                                                // update maximum of Z(k)
                                                currentZ = SIMD::max(currentZ, substEdit);
                                                currentZ = SIMD::max(currentZ, insEdit);
                                            }
                                        }

                                        /* when Z<0 & Hi-1,k=0 -> Z=0 */
                                        __mxxxi update = _mm256_cmpgt_epi8(SIMD::zero(), currentZ);
                                        __mxxxi update32_0 = _mm256_cmpeq_epi32(lastBatchRow_0[(loopJ - 1) & 1][graphLocal.adjcny_in[m]], SIMD::zero());
                                        __mxxxi update32_1 = _mm256_cmpeq_epi32(lastBatchRow_1[(loopJ - 1) & 1][graphLocal.adjcny_in[m]], SIMD::zero());
                                        __mxxxi update32_2 = _mm256_cmpeq_epi32(lastBatchRow_2[(loopJ - 1) & 1][graphLocal.adjcny_in[m]], SIMD::zero());
                                        __mxxxi update32_3 = _mm256_cmpeq_epi32(lastBatchRow_3[(loopJ - 1) & 1][graphLocal.adjcny_in[m]], SIMD::zero());
                                        __mxxxi update1 = SIMD::combine_mask(update32_0, update32_1, update32_2, update32_3);
                                        update = _mm256_and_si256(update, update1);
                                        currentZ = _mm256_blendv_epi8(currentZ, SIMD::zero(), update);
                                        /* ************************************* */

                                        // V(k)i,j = Z(k)i,j - Ui,k
                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                            currentV = SIMD::sub(currentZ, UnearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                        else
                                            currentV = SIMD::sub(currentZ, UfartherColumns[graphLocal.adjcny_in[m]][l]);

                                        // save currentV to V0
                                        V0[i] = currentV;

                                        // if u not calculated, calculate ui,j and Hi,j
                                        if (uCalculated == false)
                                        {
                                            // Ui,j = Z(k)i,j - V(k)i-1,j
                                            currentU = SIMD::sub(currentZ, VlastBatchRow[(loopJ - 1) & 1][OriginalIndex[k] + m - graphLocal.offsets_in[k]]);

                                            UnearbyColumns[k & (blockWidth - 1)][l] = currentU;
                                            // save currentU to UnearbyColumns and UfartherColumns if longHop
                                            if (withLongHopLocal[k])
                                                UfartherColumns[k][l] = currentU;

                                            // Hi,j = Hi-1,j + Ui,j
                                            // divide Ui,j
                                            __mxxxi currentU32_0, currentU32_1, currentU32_2, currentU32_3;
                                            currentU32_0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(currentU));
                                            currentU32_1 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(currentU, 1)));
                                            currentU32_2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(currentU, 1));
                                            currentU32_3 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(currentU, 3)));
                                            currentMax512_0 = SIMD::add_32(lastBatchRow_0[(loopJ - 1) & 1][k], currentU32_0);
                                            currentMax512_1 = SIMD::add_32(lastBatchRow_1[(loopJ - 1) & 1][k], currentU32_1);
                                            currentMax512_2 = SIMD::add_32(lastBatchRow_2[(loopJ - 1) & 1][k], currentU32_2);
                                            currentMax512_3 = SIMD::add_32(lastBatchRow_3[(loopJ - 1) & 1][k], currentU32_3);

                                            // u has been calculated
                                            uCalculated = true;
                                        }
                                    }
                                    lastV = 0;
                                }

                                // case3: other rows
                                else
                                {
                                    // calculate Hi,j and ui,j directly when there is no predecessor node
                                    if (graphLocal.offsets_in[k] >= graphLocal.offsets_in[k + 1])
                                    {
                                        currentMax512_0 = SIMD::zero();
                                        currentMax512_1 = SIMD::zero();
                                        currentMax512_2 = SIMD::zero();
                                        currentMax512_3 = SIMD::zero();

                                        // divide Δij
                                        __mxxxi sub512_H0, sub512_H1, sub512_H2, sub512_H3;
                                        sub512_H0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(sub512));
                                        sub512_H1 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 1)));
                                        sub512_H2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(sub512, 1));
                                        sub512_H3 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(sub512, 3)));

                                        // update currentMax512
                                        currentMax512_0 = SIMD::max_32(currentMax512_0, sub512_H0);
                                        currentMax512_1 = SIMD::max_32(currentMax512_1, sub512_H1);
                                        currentMax512_2 = SIMD::max_32(currentMax512_2, sub512_H2);
                                        currentMax512_3 = SIMD::max_32(currentMax512_3, sub512_H3);

                                        // Hi-1,j - Δdel
                                        __mxxxi delEdit_0 = SIMD::add_32(nearbyColumns_0[k & (blockWidth - 1)][l - 1], del512_32);
                                        __mxxxi delEdit_1 = SIMD::add_32(nearbyColumns_1[k & (blockWidth - 1)][l - 1], del512_32);
                                        __mxxxi delEdit_2 = SIMD::add_32(nearbyColumns_2[k & (blockWidth - 1)][l - 1], del512_32);
                                        __mxxxi delEdit_3 = SIMD::add_32(nearbyColumns_3[k & (blockWidth - 1)][l - 1], del512_32);

                                        // update currentMax512
                                        currentMax512_0 = SIMD::max_32(currentMax512_0, delEdit_0);
                                        currentMax512_1 = SIMD::max_32(currentMax512_1, delEdit_1);
                                        currentMax512_2 = SIMD::max_32(currentMax512_2, delEdit_2);
                                        currentMax512_3 = SIMD::max_32(currentMax512_3, delEdit_3);

                                        // ui,j = Hi,j - Hi-1,j
                                        // integrate four 32-bit U into one 8-bit U
                                        __mxxxi currentU32_0 = SIMD::sub_32(currentMax512_0, nearbyColumns_0[k & (blockWidth - 1)][l - 1]);
                                        __mxxxi currentU32_1 = SIMD::sub_32(currentMax512_1, nearbyColumns_1[k & (blockWidth - 1)][l - 1]);
                                        __mxxxi currentU32_2 = SIMD::sub_32(currentMax512_2, nearbyColumns_2[k & (blockWidth - 1)][l - 1]);
                                        __mxxxi currentU32_3 = SIMD::sub_32(currentMax512_3, nearbyColumns_3[k & (blockWidth - 1)][l - 1]);
                                        currentU = SIMD::combine_reg(currentU32_0, currentU32_1, currentU32_2, currentU32_3);

                                        UnearbyColumns[k & (blockWidth - 1)][l] = currentU;
                                        // save currentU to UnearbyColumns and UfartherColumns if longHop
                                        if (withLongHopLocal[k])
                                            UfartherColumns[k][l] = currentU;
                                    }

                                    bool uCalculated = false;

                                    for (size_t m = graphLocal.offsets_in[k], i = 0; m < graphLocal.offsets_in[k + 1]; m++, i++)
                                    {
                                        __mxxxi currentZ;
                                        if (lastV == 0)
                                            currentZ = SIMD::add(V0[i], del512);
                                        else
                                            currentZ = SIMD::add(V1[i], del512);

                                        for (size_t mm = graphLocal.offsets_in[k], j = 0; mm < graphLocal.offsets_in[k + 1]; mm++, j++)
                                        {
                                            // when k = k'
                                            if (mm == m)
                                            {
                                                // subStEdit = Δij
                                                __mxxxi substEdit = sub512;

                                                // insEdit = Ui,k' - Δins = Ui,k - Δins
                                                __mxxxi insEdit;
                                                if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                                    insEdit = SIMD::add(UnearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], ins512);
                                                else
                                                    insEdit = SIMD::add(UfartherColumns[graphLocal.adjcny_in[m]][l], ins512);

                                                // update maximum of Z(k)
                                                currentZ = SIMD::max(currentZ, substEdit);
                                                currentZ = SIMD::max(currentZ, insEdit);
                                            }
                                            else
                                            {
                                                // V(k)i-1,j - V(k')i-1,j
                                                __mxxxi VsubEdit;
                                                if (lastV == 0)
                                                    VsubEdit = SIMD::sub(V0[i], V0[j]);
                                                else
                                                    VsubEdit = SIMD::sub(V1[i], V1[j]);

                                                // subStEdit = V(k)i-1,j - V(k')i-1,j + Δij
                                                __mxxxi substEdit = SIMD::add(VsubEdit, sub512);

                                                // insEdit = Ui,k' + V(k)i-1,j - V(k')i-1,j - Δins
                                                __mxxxi insEdit;
                                                if (k - graphLocal.adjcny_in[mm] < this->blockWidth)
                                                    insEdit = SIMD::add(UnearbyColumns[graphLocal.adjcny_in[mm] & (blockWidth - 1)][l], VsubEdit);
                                                else
                                                    insEdit = SIMD::add(UfartherColumns[graphLocal.adjcny_in[mm]][l], VsubEdit);
                                                insEdit = SIMD::add(insEdit, ins512);

                                                // update maximum of Z(k)
                                                currentZ = SIMD::max(currentZ, substEdit);
                                                currentZ = SIMD::max(currentZ, insEdit);
                                            }
                                        }

                                        /* when Z < 0 & Hi-1,k = 0 --> Z=0 */
                                        __mxxxi H_0, H_1, H_2, H_3;
                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                        {
                                            H_0 = nearbyColumns_0[graphLocal.adjcny_in[m] & (blockWidth - 1)][l - 1];
                                            H_1 = nearbyColumns_1[graphLocal.adjcny_in[m] & (blockWidth - 1)][l - 1];
                                            H_2 = nearbyColumns_2[graphLocal.adjcny_in[m] & (blockWidth - 1)][l - 1];
                                            H_3 = nearbyColumns_3[graphLocal.adjcny_in[m] & (blockWidth - 1)][l - 1];
                                        }
                                        else
                                        {
                                            H_0 = fartherColumns_0[graphLocal.adjcny_in[m]][l - 1];
                                            H_1 = fartherColumns_1[graphLocal.adjcny_in[m]][l - 1];
                                            H_2 = fartherColumns_2[graphLocal.adjcny_in[m]][l - 1];
                                            H_3 = fartherColumns_3[graphLocal.adjcny_in[m]][l - 1];
                                        }
                                        __mxxxi update = _mm256_cmpgt_epi8(SIMD::zero(), currentZ);
                                        __mxxxi update32_0 = _mm256_cmpeq_epi32(H_0, SIMD::zero());
                                        __mxxxi update32_1 = _mm256_cmpeq_epi32(H_1, SIMD::zero());
                                        __mxxxi update32_2 = _mm256_cmpeq_epi32(H_2, SIMD::zero());
                                        __mxxxi update32_3 = _mm256_cmpeq_epi32(H_3, SIMD::zero());
                                        __mxxxi update1 = SIMD::combine_mask(update32_0, update32_1, update32_2, update32_3);
                                        update = _mm256_and_si256(update, update1);
                                        currentZ = _mm256_blendv_epi8(currentZ, SIMD::zero(), update);
                                        /* ************************************* */

                                        // V(k)i,j = Z(k)i,j - Ui,k
                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                            currentV = SIMD::sub(currentZ, UnearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l]);
                                        else
                                            currentV = SIMD::sub(currentZ, UfartherColumns[graphLocal.adjcny_in[m]][l]);

                                        if (lastV == 0)
                                            V1[i] = currentV;
                                        else
                                            V0[i] = currentV;
                                        // save last row of current batch to VlastBatchRow
                                        if (l == this->blockHeight - 1)
                                            VlastBatchRow[loopJ & 1][OriginalIndex[k] + m - graphLocal.offsets_in[k]] = currentV;

                                        // if u not calculated, calculate ui,j and Hi,j
                                        if (uCalculated == false)
                                        {
                                            // Ui,j = Z(k)i,j - V(k)i-1,j
                                            if (lastV == 0)
                                                currentU = SIMD::sub(currentZ, V0[i]);
                                            else
                                                currentU = SIMD::sub(currentZ, V1[i]);

                                            UnearbyColumns[k & (blockWidth - 1)][l] = currentU;
                                            // save currentU to UnearbyColumns and UfartherColumns if longHop
                                            if (withLongHopLocal[k])
                                                UfartherColumns[k][l] = currentU;

                                            // Hi,j = Hi-1,j + Ui,j
                                            // divide Ui,j
                                            __mxxxi currentU32_0, currentU32_1, currentU32_2, currentU32_3;
                                            currentU32_0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(currentU));
                                            currentU32_1 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(currentU, 1)));
                                            currentU32_2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(currentU, 1));
                                            currentU32_3 = _mm256_cvtepi8_epi32(_mm_set1_epi64x(_mm256_extract_epi64(currentU, 3)));
                                            currentMax512_0 = SIMD::add_32(nearbyColumns_0[k & (blockWidth - 1)][l - 1], currentU32_0);
                                            currentMax512_1 = SIMD::add_32(nearbyColumns_1[k & (blockWidth - 1)][l - 1], currentU32_1);
                                            currentMax512_2 = SIMD::add_32(nearbyColumns_2[k & (blockWidth - 1)][l - 1], currentU32_2);
                                            currentMax512_3 = SIMD::add_32(nearbyColumns_3[k & (blockWidth - 1)][l - 1], currentU32_3);

                                            // U has been calculated
                                            uCalculated = true;
                                        }
                                    }
                                    lastV = (lastV + 1) & 1;
                                }

                                // update bestScores512
                                bestScores512_0 = SIMD::max_32(currentMax512_0, bestScores512_0);
                                bestScores512_1 = SIMD::max_32(currentMax512_1, bestScores512_1);
                                bestScores512_2 = SIMD::max_32(currentMax512_2, bestScores512_2);
                                bestScores512_3 = SIMD::max_32(currentMax512_3, bestScores512_3);

                                // which channels have updated best scores
                                auto updated_0 = SIMD::cmpeq_32(currentMax512_0, bestScores512_0);
                                auto updated_1 = SIMD::cmpeq_32(currentMax512_1, bestScores512_1);
                                auto updated_2 = SIMD::cmpeq_32(currentMax512_2, bestScores512_2);
                                auto updated_3 = SIMD::cmpeq_32(currentMax512_3, bestScores512_3);

                                // update bestRows512
                                bestRows512_0 = SIMD::mask_set1_32(bestRows512_0, updated_0, (int32_t)(j + l));
                                bestRows512_1 = SIMD::mask_set1_32(bestRows512_1, updated_1, (int32_t)(j + l));
                                bestRows512_2 = SIMD::mask_set1_32(bestRows512_2, updated_2, (int32_t)(j + l));
                                bestRows512_3 = SIMD::mask_set1_32(bestRows512_3, updated_3, (int32_t)(j + l));

                                // update bestCols512
                                bestCols512_0 = SIMD::mask_set1_32(bestCols512_0, updated_0, (int32_t)k);
                                bestCols512_1 = SIMD::mask_set1_32(bestCols512_1, updated_1, (int32_t)k);
                                bestCols512_2 = SIMD::mask_set1_32(bestCols512_2, updated_2, (int32_t)k);
                                bestCols512_3 = SIMD::mask_set1_32(bestCols512_3, updated_3, (int32_t)k);

                                // save currentMax512 to nearbyColumns
                                nearbyColumns_0[k & (blockWidth - 1)][l] = currentMax512_0;
                                nearbyColumns_1[k & (blockWidth - 1)][l] = currentMax512_1;
                                nearbyColumns_2[k & (blockWidth - 1)][l] = currentMax512_2;
                                nearbyColumns_3[k & (blockWidth - 1)][l] = currentMax512_3;

                                // save to fartherColumns if longHop
                                if (withLongHopLocal[k])
                                {
                                    fartherColumns_0[k][l] = currentMax512_0;
                                    fartherColumns_1[k][l] = currentMax512_1;
                                    fartherColumns_2[k][l] = currentMax512_2;
                                    fartherColumns_3[k][l] = currentMax512_3;
                                }
                            }

                            // save last row of current batch to lastBatchRow
                            lastBatchRow_0[loopJ & 1][k] = currentMax512_0;
                            lastBatchRow_1[loopJ & 1][k] = currentMax512_1;
                            lastBatchRow_2[loopJ & 1][k] = currentMax512_2;
                            lastBatchRow_3[loopJ & 1][k] = currentMax512_3;

                        } // row-major computation ended
                    } // DP matrix computation ended

                    // store bestScores512 to bestScores
                    {
                        bestScores[4 * i + 0] = bestScores512_0;
                        bestScores[4 * i + 1] = bestScores512_1;
                        bestScores[4 * i + 2] = bestScores512_2;
                        bestScores[4 * i + 3] = bestScores512_3;
                        bestRows[4 * i + 0] = bestRows512_0;
                        bestRows[4 * i + 1] = bestRows512_1;
                        bestRows[4 * i + 2] = bestRows512_2;
                        bestRows[4 * i + 3] = bestRows512_3;
                        bestCols[4 * i + 0] = bestCols512_0;
                        bestCols[4 * i + 1] = bestCols512_1;
                        bestCols[4 * i + 2] = bestCols512_2;
                        bestCols[4 * i + 3] = bestCols512_3;
                    }
                }

                // the end time of each thread
                threadTimings[omp_get_thread_num()] = omp_get_wtime() - threadTimings[omp_get_thread_num()];

            } // omp parallel ended

            std::cout << "TIMER, hdga::alignToDAGLocal_Phase1_vectorized_difference"
                      << " (precision= " << sizeof(typename SIMD::type) << " bytes)\n";
            //           << ", individual thread timings (s) : "
            //           << printStats(threadTimings) << "\n";

            // for (int i = 0; i < omp_get_max_threads(); i++)
            // {
            //     std::cout << "thread " << i << " read_base:" << threadReadLengths[i] << ", time:" << threadTimings[i] << "\n";
            // }

#ifdef VTUNE_SUPPORT
            __itt_pause();
#endif
        }
    };

    /**
     * @brief   Supports phase 1 DP in forward direction
     */
    template <typename SIMD>
    class Phase1_Vectorized
    {
    private:
        // reference graph
        const CSR_char_container &graph;

        // pre-compute which graph vertices are connected with hop longer
        // than 'blockWidth'
        std::vector<bool> withLongHop;

        // for converting input reads into SOA to enable vectorization
        std::vector<char> readSetSOA;

        // cumulative read batch sizes
        std::vector<size_t> readSetSOAPrefixSum;

        // input reads
        const std::vector<std::string> &readSet;

        // read lengths in their 'sorted order'
        std::vector<size_t> sortedReadLengths;

        // sorted permutation order of input reads
        std::vector<size_t> sortedReadOrder;

        // input parameters (e.g., scoring scheme)
        const Parameters &parameters;

    public:
        // small temporary storage buffer for DP scores
        // should be a power of 2
        static constexpr size_t blockWidth = 8;

        // process these many vertical cells in a go
        // should be a power of 2
        static constexpr size_t blockHeight = 16;

        /**
         * @brief                   public constructor
         * @param[in]   readSet     vector of input query sequences to align
         * @param[in]   g           input reference graph
         */
        Phase1_Vectorized(const std::vector<std::string> &readSet,
                          const CSR_char_container &g,
                          const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->sortReadsForLoadBalance();
            this->convertToSOA();
            this->computeLongHops();
        };

        /**
         * @brief                                 wrapper function for phase 1 routine
         * @param[out]  outputBestScoreVector     vector to keep value and location of best scores,
         *                                        vector size is same as count of the reads
         * @note                                  this class won't care about rev. complement of sequences
         */
        template <typename Vec>
        void alignToDAGLocal_Phase1_vectorized_wrapper(Vec &outputBestScoreVector) const
        {
            assert(outputBestScoreVector.size() == readSet.size());

            std::size_t countReadBatches = std::ceil(readSet.size() * 1.0 / SIMD::numSeqs);

            // column location value requires int32_t type, so may need >1 register per read batch
            constexpr size_t colValuesPerRegister = SIMD_REG_SIZE / (8 * sizeof(int32_t));
            constexpr size_t colRegistersCountPerBatch = SIMD::numSeqs / colValuesPerRegister;

            static_assert(colRegistersCountPerBatch == 1 ||
                              colRegistersCountPerBatch == 2 ||
                              colRegistersCountPerBatch == 4,
                          "has to be either 1, 2 or 4");

            // modified containers to hold best-score info
            // vector elements aligned to 64 byte boundaries
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>> _bestScoreVector(countReadBatches);
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>
                _bestScoreColVector(countReadBatches * colRegistersCountPerBatch);
            std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>> _bestScoreRowVector(countReadBatches);

            // execute the alignment routine
            this->alignToDAGLocal_Phase1_vectorized(_bestScoreVector, _bestScoreColVector, _bestScoreRowVector);

            // when debugging for low-precision
#ifdef DEBUG
            for (auto &e : _bestScoreVector)
                simdUtils<typename SIMD::type>::print_avx_num(e);

            for (auto &e : _bestScoreRowVector)
                simdUtils<typename SIMD::type>::print_avx_num(e);

            for (auto &e : _bestScoreColVector)
                simdUtils<int32_t>::print_avx_num(e);
#endif

            // parse best scores from vector registers
            std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> storeScores(SIMD::numSeqs);
            std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> storeRows(SIMD::numSeqs);
            std::vector<int32_t, aligned_alloc<int32_t, 64>> storeCols(SIMD::numSeqs);

            for (size_t i = 0; i < countReadBatches; i++)
            {
                SIMD::store((__mxxxi *)storeScores.data(), _bestScoreVector[i]);
                SIMD::store((__mxxxi *)storeRows.data(), _bestScoreRowVector[i]);

                for (size_t j = 0; j < colRegistersCountPerBatch; j++)
                {
                    SIMD::store((__mxxxi *)&storeCols[j * colValuesPerRegister],
                                _bestScoreColVector[i * colRegistersCountPerBatch + j]);
                }

                for (size_t j = 0; j < SIMD::numSeqs; j++)
                {
                    if (i * SIMD::numSeqs + j < readSet.size())
                    {
                        auto originalReadId = sortedReadOrder[i * SIMD::numSeqs + j];

                        outputBestScoreVector[originalReadId].score = storeScores[j];
                        outputBestScoreVector[originalReadId].refColumnEnd = storeCols[j];
                        outputBestScoreVector[originalReadId].qryRowEnd = storeRows[j];

#ifdef DEBUG
                        std::cout << "INFO, hdga::Phase1_Vectorized::alignToDAGLocal_Phase1_vectorized_wrapper, read # " << originalReadId << ",  score = " << (int)storeScores[j] << ", qryRowEnd = " << (int)storeRows[j] << ", refColumnEnd = " << (int)storeCols[j] << "\n";
#endif
                    }
                }
            }
        }

    private:
        /**
         * @brief       compute the sorted order of sequences for load balancing
         * @details     sorting is done in decreasing length order
         */
        void sortReadsForLoadBalance()
        {
            typedef std::pair<size_t, size_t> pair_t;

            // vector of tuples of length and index of reads
            std::vector<pair_t> lengthTuples;

            for (size_t i = 0; i < this->readSet.size(); i++)
                lengthTuples.emplace_back(readSet[i].length(), i);

            // sort in descending order, longer reads first
            std::sort(lengthTuples.begin(), lengthTuples.end(), [](const pair_t &left, const pair_t &right)
                      { return left.first > right.first || (left.first == right.first && left.second < right.second); });

            for (auto &e : lengthTuples)
            {
                this->sortedReadLengths.push_back(e.first);
                this->sortedReadOrder.push_back(e.second);
            }
        }

        /**
         * @brief       convert input reads characters into SOA to enable vectorization
         */
        void convertToSOA()
        {
            assert(readSet.size() > 0);
            assert(sortedReadOrder.size() == readSet.size());
            assert(readSetSOA.size() == 0);

            /**
             * Requirements from the padding process
             * - each read length be a multiple of 'blockHeight'
             * - count of reads be a multiple of SIMD::numSeqs
             * - each batch of SIMD::numSeqs reads should be of equal length
             */

            // re-arrange read characters for vectorized processing;
            // SIMD::numSeqs reads at a time, sorted in decreasing order by length

            auto readCount = readSet.size();

            readSetSOAPrefixSum.push_back(0);

            for (size_t i = 0; i < readCount; i += SIMD::numSeqs)
            {
                auto batchLength = readSet[sortedReadOrder[i]].length();          // longest read in this batch
                batchLength += blockHeight - 1 - (batchLength - 1) % blockHeight; // round-up

                for (size_t j = 0; j < batchLength; j++)
                    for (size_t k = 0; k < SIMD::numSeqs; k++)
                        if (i + k < readCount && j < readSet[sortedReadOrder[i + k]].length())
                            readSetSOA.push_back(readSet[sortedReadOrder[i + k]][j]);
                        else
                            readSetSOA.push_back(DUMMY);

                readSetSOAPrefixSum.push_back(readSetSOA.size());
            }

            std::size_t countReadBatches = std::ceil(readSet.size() * 1.0 / SIMD::numSeqs);
            assert(readSetSOAPrefixSum.size() == countReadBatches + 1);
        }

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
                    if (to_pos - from_pos >= this->blockWidth)
                        this->withLongHop[from_pos] = true;
                }
            }

#ifdef DEBUG
            auto trueCount = std::count(withLongHop.begin(), withLongHop.end(), true);
            std::cout << "INFO, hdga::Phase1_Vectorized::computeLongHops, fraction of hops that are long: " << trueCount * 1.0 / graph.numVertices << "\n";
#endif
        }

        /**
         * @brief                         execute first phase of alignment i.e. compute DP and
         *                                find locations of the best alignment of each read
         * @param[out]  bestScores        best DP scores of reads
         * @param[out]  bestCols          columns where best alignment ends (for traceback later)
         * @param[out]  bestRows          rows where best alignment ends
         */
        template <typename Vec>
        void alignToDAGLocal_Phase1_vectorized(Vec &bestScores, Vec &bestCols, Vec &bestRows) const
        {
            std::size_t readCount = readSet.size();
            std::size_t countReadBatches = std::ceil(readCount * 1.0 / SIMD::numSeqs);

            // column location value requires int32_t type, so may need >1 register per read batch
            constexpr size_t colValuesPerRegister = SIMD_REG_SIZE / (8 * sizeof(int32_t));
            constexpr size_t colRegistersCountPerBatch = SIMD::numSeqs / colValuesPerRegister;

            // few checks
            assert(bestScores.size() == countReadBatches);
            assert(bestCols.size() == countReadBatches * colRegistersCountPerBatch);
            assert(bestRows.size() == countReadBatches);

#ifdef VTUNE_SUPPORT
            __itt_resume();
#endif

            // init best score vector to zero bits
            std::fill(bestScores.begin(), bestScores.end(), SIMD::zero());
            std::fill(bestCols.begin(), bestCols.end(), SIMD::zero());
            std::fill(bestRows.begin(), bestRows.end(), SIMD::zero());

            // init score simd vectors
            __mxxxi match512 = SIMD::set1((typename SIMD::type)parameters.match);
            __mxxxi mismatch512 = SIMD::set1((typename SIMD::type) - 1 * parameters.mismatch);
            __mxxxi del512 = SIMD::set1((typename SIMD::type) - 1 * parameters.del);
            __mxxxi ins512 = SIMD::set1((typename SIMD::type) - 1 * parameters.ins);

            std::vector<double> threadTimings(omp_get_max_threads(), 0);

            // copy graph as function variable for faster access
            const CSR_char_container graphLocal = this->graph;
            const std::vector<bool> withLongHopLocal = withLongHop;

#pragma omp parallel
            {
#pragma omp barrier
                threadTimings[omp_get_thread_num()] = omp_get_wtime();

                // type def. for memory-aligned vector allocation for SIMD instructions
                using AlignedVecType = std::vector<__mxxxi, aligned_alloc<__mxxxi, 64>>;

                // 2D buffer to save selected columns (associated with long hops) of DP matrix
                std::size_t countLongHops = std::count(withLongHopLocal.begin(), withLongHopLocal.end(), true);
                AlignedVecType fartherColumnsBuffer(countLongHops * this->blockHeight);

                // pointer array for convenient data access in 2D buffer
                std::vector<__mxxxi *> fartherColumns(graphLocal.numVertices);
                {
                    size_t j = 0;

                    for (int32_t i = 0; i < graphLocal.numVertices; i++)
                        if (withLongHopLocal[i])
                            fartherColumns[i] = &fartherColumnsBuffer[(j++) * this->blockHeight];
                }

                // buffer to save neighboring column scores
                AlignedVecType nearbyColumnsBuffer(this->blockWidth * this->blockHeight);

                // for convenient access to 2D buffer
                std::vector<__mxxxi *> nearbyColumns(this->blockWidth);
                {
                    for (std::size_t i = 0; i < this->blockWidth; i++)
                        nearbyColumns[i] = &nearbyColumnsBuffer[i * this->blockHeight];
                }

                // buffer to save scores of last row in each iteration
                // one row for writing and one for reading
                AlignedVecType lastBatchRowBuffer(2 * graphLocal.numVertices);

                // for convenient access to 2D buffer
                std::vector<__mxxxi *> lastBatchRow(2);
                {
                    lastBatchRow[0] = &lastBatchRowBuffer[0];
                    lastBatchRow[1] = &lastBatchRowBuffer[graphLocal.numVertices];
                }

                // buffer to save read charactes for innermost loop
                std::vector<typename SIMD::type, aligned_alloc<typename SIMD::type, 64>> readCharsInt(SIMD::numSeqs * this->blockHeight);

                // process SIMD::numSeqs reads in a single iteration
#pragma omp for schedule(dynamic) nowait
                for (size_t i = 0; i < countReadBatches; i++)
                {
                    // std::cout << "ReadBatch "<< i <<" begins.\n";
                    __mxxxi bestScores512 = SIMD::zero();
                    __mxxxi bestRows512 = SIMD::zero();

                    // we may need at most 4 registers to save column for each batch (depending on SIMD::type)
                    __mxxxi bestCols512_0 = SIMD::zero();
                    __mxxxi bestCols512_1 = SIMD::zero();
                    __mxxxi bestCols512_2 = SIMD::zero();
                    __mxxxi bestCols512_3 = SIMD::zero();

                    // reset DP 'lastBatchRow' buffer
                    std::fill(lastBatchRowBuffer.begin(), lastBatchRowBuffer.end(), SIMD::zero());

                    int32_t qryBatchLength = readSet[sortedReadOrder[i * SIMD::numSeqs]].length();      // longest read in this batch
                    qryBatchLength += this->blockHeight - 1 - (qryBatchLength - 1) % this->blockHeight; // round-up

                    // iterate over read length (process more than 1 characters in batch)
                    for (int32_t j = 0; j < qryBatchLength; j += this->blockHeight)
                    {
                        // loop counter
                        size_t loopJ = j / (this->blockHeight);

                        // convert read character to int32_t
                        for (int32_t k = 0; k < SIMD::numSeqs * this->blockHeight; k++)
                        {
                            readCharsInt[k] = readSetSOA[readSetSOAPrefixSum[i] + j * SIMD::numSeqs + k];
                        }

                        // iterate over characters in reference graph
                        for (int32_t k = 0; k < graphLocal.numVertices; k++)
                        {
                            // current reference character
                            __mxxxi graphChar = SIMD::set1((typename SIMD::type)graphLocal.vertex_label[k]);

                            // current best score, init to 0
                            __mxxxi currentMax512;

                            // iterate over 'blockHeight' read characters
                            for (size_t l = 0; l < this->blockHeight; l++)
                            {
                                // load read characters
                                __mxxxi readChars = SIMD::load((const __mxxxi *)&readCharsInt[l * SIMD::numSeqs]);

                                // current best score, init to 0
                                currentMax512 = SIMD::zero();

                                // see if query and reference character match
                                auto compareChar = SIMD::cmpeq(readChars, graphChar);
                                __mxxxi sub512 = SIMD::blend(compareChar, mismatch512, match512);

                                // match-mismatch edit
                                currentMax512 = SIMD::max(currentMax512, sub512); // local alignment can also start with a match at this char

                                // iterate over graph neighbors
                                // which buffers to access depends on the value of 'l'
                                if (l == 0)
                                {
                                    for (size_t m = graphLocal.offsets_in[k]; m < graphLocal.offsets_in[k + 1]; m++)
                                    {
                                        // paths with match mismatch edit
                                        __mxxxi substEdit = SIMD::add(lastBatchRow[(loopJ - 1) & 1][graphLocal.adjcny_in[m]], sub512);
                                        currentMax512 = SIMD::max(currentMax512, substEdit);

                                        // paths with deletion edit
                                        __mxxxi delEdit;

                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                            delEdit = SIMD::add(nearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], del512);
                                        else
                                            delEdit = SIMD::add(fartherColumns[graphLocal.adjcny_in[m]][l], del512);

                                        currentMax512 = SIMD::max(currentMax512, delEdit);
                                    }

                                    // insertion edit
                                    __mxxxi insEdit = SIMD::add(lastBatchRow[(loopJ - 1) & 1][k], ins512);
                                    currentMax512 = SIMD::max(currentMax512, insEdit);
                                }
                                else
                                {
                                    for (size_t m = graphLocal.offsets_in[k]; m < graphLocal.offsets_in[k + 1]; m++)
                                    {
                                        // paths with match mismatch edit
                                        __mxxxi substEdit;

                                        // paths with deletion edit
                                        __mxxxi delEdit;

                                        if (k - graphLocal.adjcny_in[m] < this->blockWidth)
                                        {
                                            substEdit = SIMD::add(nearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l - 1], sub512);
                                            delEdit = SIMD::add(nearbyColumns[graphLocal.adjcny_in[m] & (blockWidth - 1)][l], del512);
                                        }
                                        else
                                        {
                                            substEdit = SIMD::add(fartherColumns[graphLocal.adjcny_in[m]][l - 1], sub512);
                                            delEdit = SIMD::add(fartherColumns[graphLocal.adjcny_in[m]][l], del512);
                                        }

                                        currentMax512 = SIMD::max(currentMax512, substEdit);
                                        currentMax512 = SIMD::max(currentMax512, delEdit);
                                    }

                                    // insertion edit
                                    __mxxxi insEdit = SIMD::add(nearbyColumns[k & (blockWidth - 1)][l - 1], ins512);
                                    currentMax512 = SIMD::max(currentMax512, insEdit);
                                }

                                // update best score observed yet
                                bestScores512 = SIMD::max(currentMax512, bestScores512);

                                // on which lanes is the best score updated
                                auto updated = SIMD::cmpeq(currentMax512, bestScores512);

                                // update row and column values accordingly
                                bestRows512 = SIMD::mask_set1(bestRows512, updated, (typename SIMD::type)(j + l));
                                SIMD::update_cols(bestCols512_0, bestCols512_1, bestCols512_2, bestCols512_3, (int32_t)k, updated);

                                // save current score in small buffer
                                nearbyColumns[k & (blockWidth - 1)][l] = currentMax512;

                                // save current score in large buffer if connected thru long hop
                                if (withLongHopLocal[k])
                                    fartherColumns[k][l] = currentMax512;
                            }

                            // save last score for next row-wise iteration
                            lastBatchRow[loopJ & 1][k] = currentMax512;

                        } // end of row computation
                    } // end of DP

                    bestScores[i] = bestScores512;
                    bestRows[i] = bestRows512;
                    {
                        // storing best columns requires extra work
                        if (colRegistersCountPerBatch == 1)
                        {
                            bestCols[1 * i + 0] = bestCols512_0;
                        }
                        else if (colRegistersCountPerBatch == 2)
                        {
                            bestCols[2 * i + 0] = bestCols512_0;
                            bestCols[2 * i + 1] = bestCols512_1;
                        }
                        else if (colRegistersCountPerBatch == 4)
                        {
                            bestCols[4 * i + 0] = bestCols512_0;
                            bestCols[4 * i + 1] = bestCols512_1;
                            bestCols[4 * i + 2] = bestCols512_2;
                            bestCols[4 * i + 3] = bestCols512_3;
                        }
                    }

                } // all reads done

                threadTimings[omp_get_thread_num()] = omp_get_wtime() - threadTimings[omp_get_thread_num()];

            } // end of omp parallel

#ifdef VTUNE_SUPPORT
            __itt_pause();
#endif
        }
    };
}

#endif
