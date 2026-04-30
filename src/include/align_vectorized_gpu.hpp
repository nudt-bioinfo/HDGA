/**
 * @file    align_vectorized_gpu.hpp
 * @brief   Host-side wrappers for GPU alignment kernels
 */

#ifndef ALIGN_VECTORIZED_GPU_HPP
#define ALIGN_VECTORIZED_GPU_HPP

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include "csr_char.hpp"
#include "base_types.hpp"

namespace hdga
{
    // --- CUDA Kernel Launchers (declared in align_gpu_kernel.cu) ---

    extern "C" void launchAlignKernel_Phase1_int8(
        const char *readSetSOA_d, const size_t *readSetSOAPrefixSum_d, const size_t *sortedReadOrder_d, const size_t *sortedReadLengths_d,
        const char *graphVertexLabel_d, const int32_t *graphAdjcnyIn_d, const int32_t *graphOffsetsIn_d, const bool *withLongHop_d,
        int32_t *bestScores_d, int32_t *bestCols_d, int32_t *bestRows_d,
        int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
        size_t readCount, int32_t numVertices);

    extern "C" void launchAlignKernel_Phase1_int16(
        const char *readSetSOA_d, const size_t *readSetSOAPrefixSum_d, const size_t *sortedReadOrder_d, const size_t *sortedReadLengths_d,
        const char *graphVertexLabel_d, const int32_t *graphAdjcnyIn_d, const int32_t *graphOffsetsIn_d, const bool *withLongHop_d,
        int32_t *bestScores_d, int32_t *bestCols_d, int32_t *bestRows_d,
        int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
        size_t readCount, int32_t numVertices);

    extern "C" void launchAlignKernel_Phase2_int8(
        const char *readSetSOA_d, const size_t *readSetSOAPrefixSum_d, const size_t *sortedReadOrder_d, const size_t *sortedReadLengths_d,
        const char *graphVertexLabel_d, const int32_t *graphAdjcnyOut_d, const int32_t *graphOffsetsOut_d,
        const int32_t *phase1_bestCols_d, const int32_t *phase1_bestRows_d, const int32_t *phase1_bestScores_d,
        int32_t *bestScores_d, int32_t *bestCols_d, int32_t *bestRows_d,
        int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
        size_t readCount, int32_t numVertices);

    extern "C" void launchAlignKernel_Phase2_int16(
        const char *readSetSOA_d, const size_t *readSetSOAPrefixSum_d, const size_t *sortedReadOrder_d, const size_t *sortedReadLengths_d,
        const char *graphVertexLabel_d, const int32_t *graphAdjcnyOut_d, const int32_t *graphOffsetsOut_d,
        const int32_t *phase1_bestCols_d, const int32_t *phase1_bestRows_d, const int32_t *phase1_bestScores_d,
        int32_t *bestScores_d, int32_t *bestCols_d, int32_t *bestRows_d,
        int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
        size_t readCount, int32_t numVertices);

    // --- GPU Alignment Wrapper Classes ---

    template <typename T>
    class Phase1_Vectorized_GPU
    {
    private:
        const CSR_char_container &graph;
        const std::vector<std::string> &readSet;
        const Parameters &parameters;

        // Internal data structures for SOA (Structure of Arrays) format
        std::vector<char> readSetSOA;
        std::vector<size_t> readSetSOAPrefixSum;
        std::vector<size_t> sortedReadLengths;
        std::vector<size_t> sortedReadOrder;
        std::vector<bool> withLongHop;

        static constexpr size_t blockHeight = 16;

        // === OPTIMIZATION: GPU Memory Pool ===
        // 设备端指针缓存
        mutable int32_t *d_bestScores = nullptr;
        mutable int32_t *d_bestCols = nullptr;
        mutable int32_t *d_bestRows = nullptr;
        mutable char *d_readSetSOA = nullptr;
        mutable size_t *d_readSetSOAPrefixSum = nullptr;
        mutable size_t *d_sortedReadOrder = nullptr;
        mutable size_t *d_sortedReadLengths = nullptr;
        mutable char *d_graphVertexLabel = nullptr;
        mutable int32_t *d_graphAdjcnyIn = nullptr;
        mutable int32_t *d_graphOffsetsIn = nullptr;
        mutable bool *d_withLongHop = nullptr;

        // 缓存状态标记
        mutable bool gpu_memory_initialized = false;
        mutable size_t cached_read_count = 0;
        mutable size_t cached_graph_vertices = 0;

        // CUDA stream 用于异步操作
        mutable cudaStream_t stream;
        mutable bool stream_initialized = false;

        // 计时器
        mutable float total_kernel_time_ms = 0.0f;
        mutable int alignment_count = 0;

        // === OPTIMIZATION: Pinned memory for faster PCIe transfers ===
        mutable int32_t *h_pinned_bestScores = nullptr;
        mutable int32_t *h_pinned_bestCols = nullptr;
        mutable int32_t *h_pinned_bestRows = nullptr;
        mutable bool pinned_memory_initialized = false;

    public:
        Phase1_Vectorized_GPU(const std::vector<std::string> &readSet,
                              const CSR_char_container &g,
                              const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->sortReadsForLoadBalance();
            this->convertToSOA();
            this->computeLongHops();

            // === OPTIMIZATION: Initialize CUDA stream ===
            cudaStreamCreate(&stream);
            stream_initialized = true;
        };

        ~Phase1_Vectorized_GPU()
        {
            // === OPTIMIZATION: Cleanup GPU memory pool ===
            cleanupGPUMemory();
            if (stream_initialized)
            {
                cudaStreamDestroy(stream);
            }
        }

        // === OPTIMIZATION: Manual memory pool management ===
        void initializeGPUMemory(size_t readCount, size_t numVertices) const
        {
            if (gpu_memory_initialized &&
                cached_read_count == readCount &&
                cached_graph_vertices == numVertices)
            {
                return; // Memory already allocated for this size
            }

            // Free existing memory if any
            cleanupGPUMemory();

            // Allocate new memory
            cudaMalloc(&d_bestScores, readCount * sizeof(int32_t));
            cudaMalloc(&d_bestCols, readCount * sizeof(int32_t));
            cudaMalloc(&d_bestRows, readCount * sizeof(int32_t));
            cudaMalloc(&d_readSetSOA, readSetSOA.size() * sizeof(char));
            cudaMalloc(&d_readSetSOAPrefixSum, readSetSOAPrefixSum.size() * sizeof(size_t));
            cudaMalloc(&d_sortedReadOrder, sortedReadOrder.size() * sizeof(size_t));
            cudaMalloc(&d_sortedReadLengths, sortedReadLengths.size() * sizeof(size_t));
            cudaMalloc(&d_graphVertexLabel, graph.vertex_label.size() * sizeof(char));
            cudaMalloc(&d_graphAdjcnyIn, graph.adjcny_in.size() * sizeof(int32_t));
            cudaMalloc(&d_graphOffsetsIn, graph.offsets_in.size() * sizeof(int32_t));
            cudaMalloc(&d_withLongHop, withLongHop.size() * sizeof(bool));

            // === OPTIMIZATION: Allocate pinned memory for faster transfers ===
            cudaMallocHost(&h_pinned_bestScores, readCount * sizeof(int32_t));
            cudaMallocHost(&h_pinned_bestCols, readCount * sizeof(int32_t));
            cudaMallocHost(&h_pinned_bestRows, readCount * sizeof(int32_t));
            pinned_memory_initialized = true;

            cached_read_count = readCount;
            cached_graph_vertices = numVertices;
            gpu_memory_initialized = true;

            std::cout << "INFO: GPU memory pool initialized for " << readCount
                      << " reads and " << numVertices << " vertices" << std::endl;
        }

        void cleanupGPUMemory() const
        {
            if (!gpu_memory_initialized)
                return;

            cudaFree(d_bestScores);
            cudaFree(d_bestCols);
            cudaFree(d_bestRows);
            cudaFree(d_readSetSOA);
            cudaFree(d_readSetSOAPrefixSum);
            cudaFree(d_sortedReadOrder);
            cudaFree(d_sortedReadLengths);
            cudaFree(d_graphVertexLabel);
            cudaFree(d_graphAdjcnyIn);
            cudaFree(d_graphOffsetsIn);
            cudaFree(d_withLongHop);

            // === OPTIMIZATION: Free pinned memory ===
            if (pinned_memory_initialized)
            {
                cudaFreeHost(h_pinned_bestScores);
                cudaFreeHost(h_pinned_bestCols);
                cudaFreeHost(h_pinned_bestRows);
                pinned_memory_initialized = false;
            }

            // Reset pointers
            d_bestScores = nullptr;
            d_bestCols = nullptr;
            d_bestRows = nullptr;
            d_readSetSOA = nullptr;
            d_readSetSOAPrefixSum = nullptr;
            d_sortedReadOrder = nullptr;
            d_sortedReadLengths = nullptr;
            d_graphVertexLabel = nullptr;
            d_graphAdjcnyIn = nullptr;
            d_graphOffsetsIn = nullptr;
            d_withLongHop = nullptr;
            h_pinned_bestScores = nullptr;
            h_pinned_bestCols = nullptr;
            h_pinned_bestRows = nullptr;

            gpu_memory_initialized = false;
            cached_read_count = 0;
            cached_graph_vertices = 0;
        }

        // === OPTIMIZATION: Performance statistics ===
        void printPerformanceStats() const
        {
            if (alignment_count > 0)
            {
                float avg_time = total_kernel_time_ms / alignment_count;
                std::cout << "=== GPU Performance Statistics ===" << std::endl;
                std::cout << "Total alignments: " << alignment_count << std::endl;
                std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;
                std::cout << "Total GPU time: " << total_kernel_time_ms << " ms" << std::endl;
                std::cout << "=================================" << std::endl;
            }
        }

        void alignToDAGLocal_Phase1_vectorized_wrapper(std::vector<BestScoreInfo> &outputBestScoreVector) const
        {
            size_t readCount = readSet.size();
            std::cout << "INFO: Using GPU Phase1 alignment with CUDA" << std::endl;
            std::cout << "INFO: Processing " << readCount << " reads on GPU" << std::endl;

            // === OPTIMIZATION: Start timing ===
            auto start_time = std::chrono::high_resolution_clock::now();

            // === OPTIMIZATION: Initialize or reuse GPU memory pool ===
            initializeGPUMemory(readCount, graph.numVertices);

            // Copy data to GPU (using cached pointers)
            cudaMemcpyAsync(d_readSetSOA, readSetSOA.data(), readSetSOA.size() * sizeof(char),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_readSetSOAPrefixSum, readSetSOAPrefixSum.data(),
                            readSetSOAPrefixSum.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sortedReadOrder, sortedReadOrder.data(),
                            sortedReadOrder.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sortedReadLengths, sortedReadLengths.data(),
                            sortedReadLengths.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphVertexLabel, graph.vertex_label.data(),
                            graph.vertex_label.size() * sizeof(char), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphAdjcnyIn, graph.adjcny_in.data(),
                            graph.adjcny_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphOffsetsIn, graph.offsets_in.data(),
                            graph.offsets_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

            // withLongHop is vector<bool>, need careful copy
            std::vector<char> withLongHopChar(withLongHop.begin(), withLongHop.end());
            cudaMemcpyAsync(d_withLongHop, withLongHopChar.data(),
                            withLongHopChar.size() * sizeof(bool), cudaMemcpyHostToDevice, stream);

            // Determine bit-width needed
            size_t maxPossibleScore = 0;
            for (auto len : sortedReadLengths)
                maxPossibleScore = std::max(maxPossibleScore, len * parameters.match);

            // === OPTIMIZATION: Use optimized block size for RTX 3050 ===
            constexpr int BLOCK_SIZE = 256; // Optimized for RTX 3050

            if (maxPossibleScore <= 127)
            {
                launchAlignKernel_Phase1_int8(
                    d_readSetSOA, d_readSetSOAPrefixSum, d_sortedReadOrder, d_sortedReadLengths,
                    d_graphVertexLabel, d_graphAdjcnyIn, d_graphOffsetsIn, d_withLongHop,
                    d_bestScores, d_bestCols, d_bestRows,
                    parameters.match, parameters.mismatch, parameters.del, parameters.ins,
                    readCount, graph.numVertices);
            }
            else
            {
                launchAlignKernel_Phase1_int16(
                    d_readSetSOA, d_readSetSOAPrefixSum, d_sortedReadOrder, d_sortedReadLengths,
                    d_graphVertexLabel, d_graphAdjcnyIn, d_graphOffsetsIn, d_withLongHop,
                    d_bestScores, d_bestCols, d_bestRows,
                    parameters.match, parameters.mismatch, parameters.del, parameters.ins,
                    readCount, graph.numVertices);
            }

            // === OPTIMIZATION: Copy results using pinned memory for faster transfer ===
            cudaMemcpyAsync(h_pinned_bestScores, d_bestScores, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_pinned_bestCols, d_bestCols, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_pinned_bestRows, d_bestRows, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);

            // === OPTIMIZATION: Synchronize stream and measure time ===
            cudaStreamSynchronize(stream);

            auto end_time = std::chrono::high_resolution_clock::now();
            float kernel_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
            total_kernel_time_ms += kernel_time_ms;
            alignment_count++;

            std::cout << "INFO: GPU kernel execution time: " << kernel_time_ms << " ms" << std::endl;

            // Map results back to original read order
            for (size_t i = 0; i < readCount; i++)
            {
                outputBestScoreVector[i].score = h_pinned_bestScores[i];
                outputBestScoreVector[i].refColumnEnd = h_pinned_bestCols[i];
                outputBestScoreVector[i].qryRowEnd = h_pinned_bestRows[i];
            }

            std::cout << "INFO: GPU Phase1 alignment completed" << std::endl;
        }

    private:
        void sortReadsForLoadBalance()
        {
            typedef std::pair<size_t, size_t> pair_t;
            std::vector<pair_t> lengthTuples;
            for (size_t i = 0; i < this->readSet.size(); i++)
                lengthTuples.emplace_back(readSet[i].length(), i);
            std::sort(lengthTuples.begin(), lengthTuples.end(), [](const pair_t &left, const pair_t &right)
                      { return left.first > right.first || (left.first == right.first && left.second < right.second); });
            for (auto &e : lengthTuples)
            {
                this->sortedReadLengths.push_back(e.first);
                this->sortedReadOrder.push_back(e.second);
            }
        }

        void convertToSOA()
        {
            auto readCount = readSet.size();
            readSetSOAPrefixSum.push_back(0);
            for (size_t i = 0; i < readCount; i++)
            {
                auto batchLength = readSet[sortedReadOrder[i]].length();
                batchLength += blockHeight - 1 - (batchLength - 1) % blockHeight;
                for (size_t j = 0; j < batchLength; j++)
                {
                    if (j < readSet[sortedReadOrder[i]].length())
                        readSetSOA.push_back(readSet[sortedReadOrder[i]][j]);
                    else
                        readSetSOA.push_back('N');
                }
                readSetSOAPrefixSum.push_back(readSetSOA.size());
            }
        }

        void computeLongHops()
        {
            this->withLongHop.resize(graph.numVertices, false);
            for (int32_t i = 0; i < graph.numVertices; i++)
            {
                for (auto j = graph.offsets_in[i]; j < graph.offsets_in[i + 1]; j++)
                {
                    auto from = graph.adjcny_in[j];
                    if (i - from >= 32)
                        this->withLongHop[i] = true;
                }
            }
        }
    };

    template <typename T>
    class Phase2_Vectorized_GPU
    {
    private:
        const CSR_char_container &graph;
        const std::vector<std::string> &readSet;
        const Parameters &parameters;

        std::vector<char> readSetSOA;
        std::vector<size_t> readSetSOAPrefixSum;
        std::vector<size_t> sortedReadLengths;
        std::vector<size_t> sortedReadOrder;

        static constexpr size_t blockHeight = 16;

        // === OPTIMIZATION: GPU Memory Pool for Phase2 ===
        mutable int32_t *d_bestScores = nullptr;
        mutable int32_t *d_bestCols = nullptr;
        mutable int32_t *d_bestRows = nullptr;
        mutable int32_t *d_p1_bestCols = nullptr;
        mutable int32_t *d_p1_bestRows = nullptr;
        mutable int32_t *d_p1_bestScores = nullptr;
        mutable char *d_readSetSOA = nullptr;
        mutable size_t *d_readSetSOAPrefixSum = nullptr;
        mutable size_t *d_sortedReadOrder = nullptr;
        mutable size_t *d_sortedReadLengths = nullptr;
        mutable char *d_graphVertexLabel = nullptr;
        mutable int32_t *d_graphAdjcnyOut = nullptr;
        mutable int32_t *d_graphOffsetsOut = nullptr;

        mutable bool gpu_memory_initialized = false;
        mutable size_t cached_read_count = 0;
        mutable size_t cached_graph_vertices = 0;

        mutable cudaStream_t stream;
        mutable bool stream_initialized = false;

        mutable float total_kernel_time_ms = 0.0f;
        mutable int alignment_count = 0;

        // === OPTIMIZATION: Pinned memory buffers for faster PCIe transfer ===
        mutable int32_t *h_pinned_bestScores = nullptr;
        mutable int32_t *h_pinned_bestCols = nullptr;
        mutable int32_t *h_pinned_bestRows = nullptr;
        mutable bool pinned_memory_initialized = false;

    public:
        Phase2_Vectorized_GPU(const std::vector<std::string> &readSet,
                              const CSR_char_container &g,
                              const Parameters &p) : readSet(readSet), graph(g), parameters(p)
        {
            this->sortReadsForLoadBalance();
            this->convertToSOA();

            // === OPTIMIZATION: Initialize CUDA stream ===
            cudaStreamCreate(&stream);
            stream_initialized = true;
        };

        ~Phase2_Vectorized_GPU()
        {
            // === OPTIMIZATION: Cleanup GPU memory pool ===
            cleanupGPUMemory();
            if (stream_initialized)
            {
                cudaStreamDestroy(stream);
            }
        }

        // === OPTIMIZATION: Memory pool management ===
        void initializeGPUMemory(size_t readCount, size_t numVertices) const
        {
            if (gpu_memory_initialized &&
                cached_read_count == readCount &&
                cached_graph_vertices == numVertices)
            {
                return;
            }

            cleanupGPUMemory();

            cudaMalloc(&d_bestScores, readCount * sizeof(int32_t));
            cudaMalloc(&d_bestCols, readCount * sizeof(int32_t));
            cudaMalloc(&d_bestRows, readCount * sizeof(int32_t));
            cudaMalloc(&d_p1_bestCols, readCount * sizeof(int32_t));
            cudaMalloc(&d_p1_bestRows, readCount * sizeof(int32_t));
            cudaMalloc(&d_p1_bestScores, readCount * sizeof(int32_t));
            cudaMalloc(&d_readSetSOA, readSetSOA.size() * sizeof(char));
            cudaMalloc(&d_readSetSOAPrefixSum, readSetSOAPrefixSum.size() * sizeof(size_t));
            cudaMalloc(&d_sortedReadOrder, sortedReadOrder.size() * sizeof(size_t));
            cudaMalloc(&d_sortedReadLengths, sortedReadLengths.size() * sizeof(size_t));
            cudaMalloc(&d_graphVertexLabel, graph.vertex_label.size() * sizeof(char));
            cudaMalloc(&d_graphAdjcnyOut, graph.adjcny_out.size() * sizeof(int32_t));
            cudaMalloc(&d_graphOffsetsOut, graph.offsets_out.size() * sizeof(int32_t));

            // === OPTIMIZATION: Allocate pinned memory for faster transfers ===
            cudaMallocHost(&h_pinned_bestScores, readCount * sizeof(int32_t));
            cudaMallocHost(&h_pinned_bestCols, readCount * sizeof(int32_t));
            cudaMallocHost(&h_pinned_bestRows, readCount * sizeof(int32_t));
            pinned_memory_initialized = true;

            cached_read_count = readCount;
            cached_graph_vertices = numVertices;
            gpu_memory_initialized = true;

            std::cout << "INFO: GPU Phase2 memory pool initialized for " << readCount
                      << " reads and " << numVertices << " vertices" << std::endl;
        }

        void cleanupGPUMemory() const
        {
            if (!gpu_memory_initialized)
                return;

            cudaFree(d_bestScores);
            cudaFree(d_bestCols);
            cudaFree(d_bestRows);
            cudaFree(d_p1_bestCols);
            cudaFree(d_p1_bestRows);
            cudaFree(d_p1_bestScores);
            cudaFree(d_readSetSOA);
            cudaFree(d_readSetSOAPrefixSum);
            cudaFree(d_sortedReadOrder);
            cudaFree(d_sortedReadLengths);
            cudaFree(d_graphVertexLabel);
            cudaFree(d_graphAdjcnyOut);
            cudaFree(d_graphOffsetsOut);

            // === OPTIMIZATION: Free pinned memory ===
            if (pinned_memory_initialized)
            {
                cudaFreeHost(h_pinned_bestScores);
                cudaFreeHost(h_pinned_bestCols);
                cudaFreeHost(h_pinned_bestRows);
                pinned_memory_initialized = false;
            }

            d_bestScores = nullptr;
            d_bestCols = nullptr;
            d_bestRows = nullptr;
            d_p1_bestCols = nullptr;
            d_p1_bestRows = nullptr;
            d_p1_bestScores = nullptr;
            d_readSetSOA = nullptr;
            d_readSetSOAPrefixSum = nullptr;
            d_sortedReadOrder = nullptr;
            d_sortedReadLengths = nullptr;
            d_graphVertexLabel = nullptr;
            d_graphAdjcnyOut = nullptr;
            d_graphOffsetsOut = nullptr;
            h_pinned_bestScores = nullptr;
            h_pinned_bestCols = nullptr;
            h_pinned_bestRows = nullptr;

            gpu_memory_initialized = false;
            cached_read_count = 0;
            cached_graph_vertices = 0;
        }

        void printPerformanceStats() const
        {
            if (alignment_count > 0)
            {
                float avg_time = total_kernel_time_ms / alignment_count;
                std::cout << "=== GPU Phase2 Performance Statistics ===" << std::endl;
                std::cout << "Total alignments: " << alignment_count << std::endl;
                std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;
                std::cout << "Total GPU time: " << total_kernel_time_ms << " ms" << std::endl;
                std::cout << "======================================" << std::endl;
            }
        }

        void alignToDAGLocal_Phase2_vectorized_wrapper(std::vector<BestScoreInfo> &bestScoreVector) const
        {
            size_t readCount = readSet.size();
            std::cout << "INFO: Using GPU Phase2 backward search with CUDA" << std::endl;

            // === OPTIMIZATION: Start timing ===
            auto start_time = std::chrono::steady_clock::now();

            std::vector<int32_t> h_p1_bestCols(readCount);
            std::vector<int32_t> h_p1_bestRows(readCount);
            std::vector<int32_t> h_p1_bestScores(readCount);
            for (size_t i = 0; i < readCount; i++)
            {
                h_p1_bestCols[i] = bestScoreVector[i].refColumnEnd;
                h_p1_bestRows[i] = bestScoreVector[i].qryRowEnd;
                h_p1_bestScores[i] = bestScoreVector[i].score;
            }

            // === OPTIMIZATION: Initialize or reuse GPU memory pool ===
            initializeGPUMemory(readCount, graph.numVertices);

            // Copy data asynchronously
            cudaMemcpyAsync(d_p1_bestCols, h_p1_bestCols.data(), readCount * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_p1_bestRows, h_p1_bestRows.data(), readCount * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_p1_bestScores, h_p1_bestScores.data(), readCount * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_readSetSOA, readSetSOA.data(), readSetSOA.size() * sizeof(char),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_readSetSOAPrefixSum, readSetSOAPrefixSum.data(),
                            readSetSOAPrefixSum.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sortedReadOrder, sortedReadOrder.data(),
                            sortedReadOrder.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sortedReadLengths, sortedReadLengths.data(),
                            sortedReadLengths.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphVertexLabel, graph.vertex_label.data(),
                            graph.vertex_label.size() * sizeof(char), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphAdjcnyOut, graph.adjcny_out.data(),
                            graph.adjcny_out.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_graphOffsetsOut, graph.offsets_out.data(),
                            graph.offsets_out.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

            size_t maxPossibleScore = 0;
            for (auto len : sortedReadLengths)
                maxPossibleScore = std::max(maxPossibleScore, len * parameters.match);

            // === OPTIMIZATION: Use larger block size for better occupancy ===
            constexpr int PHASE2_BLOCK_SIZE = 512; // Increased from 256

            if (maxPossibleScore <= 127)
            {
                launchAlignKernel_Phase2_int8(
                    d_readSetSOA, d_readSetSOAPrefixSum, d_sortedReadOrder, d_sortedReadLengths,
                    d_graphVertexLabel, d_graphAdjcnyOut, d_graphOffsetsOut,
                    d_p1_bestCols, d_p1_bestRows, d_p1_bestScores,
                    d_bestScores, d_bestCols, d_bestRows,
                    parameters.match, parameters.mismatch, parameters.del, parameters.ins,
                    readCount, graph.numVertices);
            }
            else
            {
                launchAlignKernel_Phase2_int16(
                    d_readSetSOA, d_readSetSOAPrefixSum, d_sortedReadOrder, d_sortedReadLengths,
                    d_graphVertexLabel, d_graphAdjcnyOut, d_graphOffsetsOut,
                    d_p1_bestCols, d_p1_bestRows, d_p1_bestScores,
                    d_bestScores, d_bestCols, d_bestRows,
                    parameters.match, parameters.mismatch, parameters.del, parameters.ins,
                    readCount, graph.numVertices);
            }

            // === OPTIMIZATION: Copy results using pinned memory for faster transfer ===
            cudaMemcpyAsync(h_pinned_bestScores, d_bestScores, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_pinned_bestCols, d_bestCols, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_pinned_bestRows, d_bestRows, readCount * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);

            // === OPTIMIZATION: Synchronize and measure time ===
            cudaStreamSynchronize(stream);

            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            float kernel_time_ms = duration / 1000.0f * 1000.0f; // Convert to ms
            total_kernel_time_ms += kernel_time_ms;
            alignment_count++;

            std::cout << "    << [DONE ] backward computing [GPU] | Cost: " << duration / 1000.0 << " s" << std::endl;
            std::cout << "INFO: GPU Phase2 kernel time: " << kernel_time_ms << " ms" << std::endl;

            for (size_t i = 0; i < readCount; i++)
            {
                bestScoreVector[i].backward_score = h_pinned_bestScores[i];
                bestScoreVector[i].refColumnStart = h_pinned_bestCols[i];
                int32_t qryEnd = bestScoreVector[i].qryRowEnd;
                int32_t revStart = h_pinned_bestRows[i];
                int32_t qryStart = qryEnd - revStart;
                if (qryStart < 0)
                    qryStart = 0;
                bestScoreVector[i].qryRowStart = qryStart;
            }

            std::cout << "INFO: GPU Phase2 alignment completed" << std::endl;
        }

    private:
        void sortReadsForLoadBalance()
        {
            typedef std::pair<size_t, size_t> pair_t;
            std::vector<pair_t> lengthTuples;
            for (size_t i = 0; i < this->readSet.size(); i++)
                lengthTuples.emplace_back(readSet[i].length(), i);
            std::sort(lengthTuples.begin(), lengthTuples.end(), [](const pair_t &left, const pair_t &right)
                      { return left.first > right.first || (left.first == right.first && left.second < right.second); });
            for (auto &e : lengthTuples)
            {
                this->sortedReadLengths.push_back(e.first);
                this->sortedReadOrder.push_back(e.second);
            }
        }

        void convertToSOA()
        {
            auto readCount = readSet.size();
            readSetSOAPrefixSum.push_back(0);
            for (size_t i = 0; i < readCount; i++)
            {
                auto batchLength = readSet[sortedReadOrder[i]].length();
                batchLength += blockHeight - 1 - (batchLength - 1) % blockHeight;
                for (size_t j = 0; j < batchLength; j++)
                {
                    if (j < readSet[sortedReadOrder[i]].length())
                        readSetSOA.push_back(readSet[sortedReadOrder[i]][j]);
                    else
                        readSetSOA.push_back('N');
                }
                readSetSOAPrefixSum.push_back(readSetSOA.size());
            }
        }
    };
}

#endif
