/**
 * @file    align_gpu_kernel.cu
 * @brief   CUDA kernel implementation for Phase1 sequence-to-graph alignment
 * @author  GPU implementation
 * @note    本文件包含实际的CUDA kernel实现，用于GPU加速的序列到图比对
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

// ========== 常量定义 ==========
// 必须与 align_vectorized_gpu.hpp 中的定义保持一致
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

// Shared Memory 配置
// BLOCK_WIDTH * BLOCK_HEIGHT = 32 * 16 = 512
// Pad to 516 (512 + 4) to avoid bank conflicts
#define SHARED_MEM_PAD 4
#define SHARED_MEM_STRIDE (BLOCK_WIDTH * BLOCK_HEIGHT + SHARED_MEM_PAD)

// Banded Alignment 配置
#define DEFAULT_BANDWIDTH 64  // 默认带宽（条带宽度）
#define MIN_BANDWIDTH 16      // 最小带宽
#define MAX_BANDWIDTH 256     // 最大带宽

#define PHASE2_MAX_PREV_ROWS 8
#define PHASE2_MAX_BAND_CELLS (DEFAULT_BANDWIDTH * 2 + 1)
#define PHASE2_MAX_FAR_ROWS 256

// ========== CUDA Kernel实现 (Templated) ==========

/**
 * @brief   CUDA kernel for Phase1 alignment computation
 */
template <typename T>
__global__ __launch_bounds__(128, 4) void alignKernel_Phase1_GPU_Template(
    const char* __restrict__ readSetSOA_d,
    const size_t* __restrict__ readSetSOAPrefixSum_d,
    const size_t* __restrict__ sortedReadOrder_d,
    const size_t* __restrict__ sortedReadLengths_d,
    const char* __restrict__ graphVertexLabel_d,
    const int32_t* __restrict__ graphAdjcnyIn_d,
    const int32_t* __restrict__ graphOffsetsIn_d,
    const bool* __restrict__ withLongHop_d,
    int32_t* __restrict__ bestScores_d,
    int32_t* __restrict__ bestCols_d,
    int32_t* __restrict__ bestRows_d,
    int32_t match_score,
    int32_t mismatch_penalty,
    int32_t del_penalty,
    int32_t ins_penalty,
    size_t readCount,
    int32_t numVertices,
    T* __restrict__ globalLastBatchRow_d)
{
    const int readIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (readIdx >= readCount) return;

    const int32_t originalReadIdx = (int32_t)sortedReadOrder_d[readIdx];
    const size_t qryLength = sortedReadLengths_d[readIdx];
    const int32_t qryBatchLength = ((int32_t)qryLength + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT * BLOCK_HEIGHT;
    const size_t prefixSum = readSetSOAPrefixSum_d[readIdx];

    T bestScore = 0;
    int32_t bestRow = 0;
    int32_t bestCol = 0;

    T nearbyColumnsBuffer[BLOCK_WIDTH * BLOCK_HEIGHT];
    #pragma unroll
    for (int i = 0; i < BLOCK_WIDTH * BLOCK_HEIGHT; ++i) {
        nearbyColumnsBuffer[i] = 0;
    }

    for (int32_t j = 0; j < qryBatchLength; j += BLOCK_HEIGHT) {
        char batchReadChars[BLOCK_HEIGHT];
        #pragma unroll
        for (int l = 0; l < BLOCK_HEIGHT; ++l) {
            size_t readPos = prefixSum + j + l;
            if (readPos < prefixSum + qryLength)
                batchReadChars[l] = readSetSOA_d[readPos];
            else
                batchReadChars[l] = 'N';
        }

        const int32_t loopJ = j / BLOCK_HEIGHT;
        const size_t prevBatchBase = ((size_t)((loopJ - 1) & 1) * (size_t)numVertices) * readCount;
        const size_t thisBatchBase = ((size_t)(loopJ & 1) * (size_t)numVertices) * readCount;

        #pragma unroll 1
        for (int32_t k = 0; k < numVertices; ++k) {
            const char graphChar = graphVertexLabel_d[k];

            T sub_scores[BLOCK_HEIGHT];
            #pragma unroll
            for (int l = 0; l < BLOCK_HEIGHT; ++l) {
                sub_scores[l] = (batchReadChars[l] == graphChar) ? (T)match_score : (T)(-mismatch_penalty);
            }

            T currMaxArr[BLOCK_HEIGHT];
            #pragma unroll
            for (int l = 0; l < BLOCK_HEIGHT; ++l) {
                currMaxArr[l] = max((T)0, sub_scores[l]);
            }

            const int32_t start_m = graphOffsetsIn_d[k];
            const int32_t end_m = graphOffsetsIn_d[k + 1];
            const int32_t base_col = (k & (BLOCK_WIDTH - 1)) * BLOCK_HEIGHT;

            for (int32_t edge = start_m; edge < end_m; ++edge) {
                const int32_t from_pos = graphAdjcnyIn_d[edge];
                if (from_pos < 0) continue;

                const int32_t offset_col = (from_pos & (BLOCK_WIDTH - 1)) * BLOCK_HEIGHT;
                const T lastBatchRowVal = (loopJ > 0)
                    ? globalLastBatchRow_d[prevBatchBase + (size_t)from_pos * readCount + readIdx]
                    : (T)0;

                #pragma unroll
                for (int l = 0; l < BLOCK_HEIGHT; ++l) {
                    T val = currMaxArr[l];
                    if (l == 0) {
                        const T substEdit = (loopJ > 0) ? lastBatchRowVal + sub_scores[l] : sub_scores[l];
                        val = max(val, substEdit);
                    } 
                    else {
                        const T substEdit = nearbyColumnsBuffer[offset_col + l - 1] + sub_scores[l];
                        val = max(val, substEdit);
                    }
                    const T delScore = nearbyColumnsBuffer[offset_col + l] - (T)del_penalty;
                    val = max(val, delScore);
                    currMaxArr[l] = val;
                }
            }

            #pragma unroll
            for (int l = 0; l < BLOCK_HEIGHT; ++l) {
                if (l == 0) {
                    if (loopJ > 0) {
                        const T insEdit = globalLastBatchRow_d[prevBatchBase + (size_t)k * readCount + readIdx] - (T)ins_penalty;
                        currMaxArr[l] = max(currMaxArr[l], insEdit);
                    }
                } 
                else {
                    const T insEdit = currMaxArr[l - 1] - (T)ins_penalty;
                    currMaxArr[l] = max(currMaxArr[l], insEdit);
                }

                const T currentMax = max((T)0, currMaxArr[l]);
                if (currentMax >= bestScore) {
                    bestScore = currentMax;
                    bestRow = j + l;
                    bestCol = k;
                }
                nearbyColumnsBuffer[base_col + l] = currentMax;
            }

            if (loopJ >= 0) {
                const size_t rowIdx = thisBatchBase + (size_t)k * readCount + readIdx;
                globalLastBatchRow_d[rowIdx] = nearbyColumnsBuffer[base_col + BLOCK_HEIGHT - 1];
            }
        }
    }

    bestScores_d[originalReadIdx] = (int32_t)bestScore;
    bestRows_d[originalReadIdx] = bestRow;
    bestCols_d[originalReadIdx] = bestCol;
}

/**
 * @brief   CUDA kernel for Phase2 alignment computation (Backward) - Non-banded Version
 * @note    GPU 全图后向 DP (不使用条带化)
 */
template <typename T>
__device__ __forceinline__ T score_cast_i32(int32_t v)
{
    return (T)v;
}

template <>
__device__ __forceinline__ int8_t score_cast_i32<int8_t>(int32_t v)
{
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}

template <>
__device__ __forceinline__ int16_t score_cast_i32<int16_t>(int32_t v)
{
    if (v < -32768) v = -32768;
    if (v > 32767) v = 32767;
    return (int16_t)v;
}

template <typename T>
__global__ void alignKernel_Phase2_GPU_Template_Banded(
    const char* readSetSOA_d,
    const size_t* readSetSOAPrefixSum_d,
    const size_t* sortedReadOrder_d,
    const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d,
    const int32_t* graphAdjcnyOut_d,
    const int32_t* graphOffsetsOut_d,
    const int32_t* phase1_bestCols_d,
    const int32_t* phase1_bestRows_d,
    const int32_t* phase1_bestScores_d,
    int32_t* bestScores_d,
    int32_t* bestCols_d,
    int32_t* bestRows_d,
    int32_t match_score,
    int32_t mismatch_penalty,
    int32_t del_penalty,
    int32_t ins_penalty,
    size_t readCount,
    int32_t numVertices,
    int32_t* fartherRowId_d,
    int32_t* fartherRowBandStart_d,
    int32_t* fartherRowBandEnd_d,
    int32_t* fartherRowMaxLeft_d,
    T* fartherRowValues_d)
{
    const int tid = threadIdx.x;
    const int readSlot = (int)blockIdx.x;
    if ((size_t)readSlot >= readCount) return;

    const int32_t originalReadIdx = (int32_t)sortedReadOrder_d[readSlot];
    const int32_t refEnd = phase1_bestCols_d[originalReadIdx];
    const int32_t qryEnd = phase1_bestRows_d[originalReadIdx];
    const int32_t fullReadLength = (int32_t)sortedReadLengths_d[readSlot];
    const int32_t forward_bestScore = phase1_bestScores_d[originalReadIdx];

    if (refEnd < 0 || refEnd >= numVertices || qryEnd < 0 || qryEnd >= fullReadLength) {
        if (tid == 0) {
            bestScores_d[originalReadIdx] = forward_bestScore;
            bestCols_d[originalReadIdx] = refEnd;
            bestRows_d[originalReadIdx] = qryEnd;
        }
        return;
    }

    __shared__ T s_curr[PHASE2_MAX_BAND_CELLS];
    __shared__ T s_matchDelete[PHASE2_MAX_BAND_CELLS];
    __shared__ T s_nearRows[PHASE2_MAX_PREV_ROWS][PHASE2_MAX_BAND_CELLS];
    __shared__ int32_t s_nearStart[PHASE2_MAX_PREV_ROWS];
    __shared__ int32_t s_nearEnd[PHASE2_MAX_PREV_ROWS];
    __shared__ int32_t s_nearRowId[PHASE2_MAX_PREV_ROWS];
    __shared__ int32_t s_nearMaxLeft[PHASE2_MAX_PREV_ROWS];
    __shared__ int32_t s_bestScore;
    __shared__ int32_t s_bestCol;
    __shared__ int32_t s_bestRow;
    __shared__ int32_t s_done;
    __shared__ int32_t s_rowStart;
    __shared__ int32_t s_rowEnd;
    __shared__ int32_t s_rowLen;

    if (tid < PHASE2_MAX_PREV_ROWS) {
        s_nearStart[tid] = 0;
        s_nearEnd[tid] = -1;
        s_nearRowId[tid] = -1;
        s_nearMaxLeft[tid] = 0;
    }
    if (tid == 0) {
        s_bestScore = 0;
        s_bestCol = refEnd;
        s_bestRow = 0;
        s_done = 0;
    }
    __syncthreads();

    const int32_t readLen = qryEnd + 1;
    int32_t Rstart = fullReadLength - qryEnd - 1;
    if (Rstart < 0) Rstart = 0;
    const size_t prefixSum = readSetSOAPrefixSum_d[readSlot];
    const int32_t w = (int32_t)(10.0 + 0.01 * (double)readLen);
    const int32_t step = 8;

    for (int32_t i = refEnd; i >= 0; --i) {
        if (s_done) break;
        if (tid == 0) {
            int32_t Mstart = 0;
            int32_t Mend = 0;
            if (i != refEnd) {
                int32_t Pleft = INT32_MAX;
                int32_t Pright = INT32_MIN;
                bool have_pre = false;
                const int32_t rowOffStart = graphOffsetsOut_d[i];
                const int32_t rowOffEnd = graphOffsetsOut_d[i + 1];
                for (int32_t e = rowOffStart; e < rowOffEnd; ++e) {
                    int32_t m = graphAdjcnyOut_d[e];
                    if (m < 0 || m > refEnd) continue;
                    int32_t preMaxLeft = 0;
                    bool found = false;
                    int32_t distance = m - i;
                    if (distance > 0 && distance < PHASE2_MAX_PREV_ROWS) {
                        int32_t nearSlot = m % PHASE2_MAX_PREV_ROWS;
                        if (s_nearRowId[nearSlot] == m) {
                            preMaxLeft = s_nearMaxLeft[nearSlot];
                            found = true;
                        }
                    }
                    if (!found) {
                        size_t base = (size_t)readSlot * (size_t)PHASE2_MAX_FAR_ROWS;
                        size_t farSlot = base + (size_t)(m % PHASE2_MAX_FAR_ROWS);
                        if (fartherRowId_d[farSlot] == m) {
                            preMaxLeft = fartherRowMaxLeft_d[farSlot];
                            found = true;
                        }
                    }
                    if (!found) continue;
                    have_pre = true;
                    if (preMaxLeft < Pleft) Pleft = preMaxLeft;
                    if (preMaxLeft > Pright) Pright = preMaxLeft;
                }
                if (have_pre) {
                    Mstart = Pleft + 1;
                    Mend = Pright + 1;
                }
            }
            int32_t Bstart0 = (Mstart - w > 0) ? (Mstart - w) : 0;
            int32_t Bend0 = (Mend + w < readLen - 1) ? (Mend + w) : (readLen - 1);
            if (Bstart0 >= Bend0) {
                s_rowStart = 0;
                s_rowEnd = -1;
                s_rowLen = 0;
            } 
            else {
                int32_t Bstart = (Bstart0 / step) * step;
                int32_t Bend = ((Bend0 + step) / step) * step - 1;
                if (Bend < Bstart) {
                    s_rowStart = 0;
                    s_rowEnd = -1;
                    s_rowLen = 0;
                } 
                else {
                    if (Bend - Bstart + 1 > PHASE2_MAX_BAND_CELLS) {
                        Bend = Bstart + PHASE2_MAX_BAND_CELLS - 1;
                    }
                    s_rowStart = Bstart;
                    s_rowEnd = Bend;
                    s_rowLen = Bend - Bstart + 1;
                }
            }
        }
        __syncthreads();
        if (s_rowLen <= 0) continue;
        const int32_t rowStart = s_rowStart;
        const int32_t rowEnd = s_rowEnd;
        const int32_t rowLen = s_rowLen;
        const char graphChar = graphVertexLabel_d[i];
        const int32_t offStart = graphOffsetsOut_d[i];
        const int32_t offEnd = graphOffsetsOut_d[i + 1];
        for (int32_t local = tid; local < rowLen; local += blockDim.x) {
            int32_t j = rowStart + local;
            size_t readPos = prefixSum + (size_t)Rstart + (size_t)j;
            char readChar = (readPos < prefixSum + (size_t)fullReadLength) ? readSetSOA_d[readPos] : 'N';
            int32_t matchScore = (readChar == graphChar) ? match_score : -mismatch_penalty;
            int32_t localBest = matchScore;
            if (localBest < 0) localBest = 0;
            for (int32_t e = offStart; e < offEnd; ++e) {
                int32_t m = graphAdjcnyOut_d[e];
                if (m < 0 || m > refEnd) continue;
                int32_t preStart;
                int32_t preEnd;
                bool useNear = false;
                int32_t distance = m - i;
                if (distance > 0 && distance < PHASE2_MAX_PREV_ROWS) {
                    int32_t nearSlot = m % PHASE2_MAX_PREV_ROWS;
                    if (s_nearRowId[nearSlot] == m) {
                        useNear = true;
                        preStart = s_nearStart[nearSlot];
                        preEnd = s_nearEnd[nearSlot];
                    }
                }
                size_t preRowBase = (size_t)readSlot * (size_t)PHASE2_MAX_FAR_ROWS;
                if (!useNear) {
                    size_t farSlot = preRowBase + (size_t)(m % PHASE2_MAX_FAR_ROWS);
                    int32_t rowId = fartherRowId_d[farSlot];
                    if (rowId != m) {
                        preStart = 1;
                        preEnd = 0;
                    } 
                    else {
                        preStart = fartherRowBandStart_d[farSlot];
                        preEnd = fartherRowBandEnd_d[farSlot];
                    }
                }
                if (preEnd < preStart) continue;
                if (j >= preStart && j <= preEnd) {
                    int32_t idx = j - preStart;
                    int32_t preVal;
                    if (useNear) {
                        int32_t nearSlot = m % PHASE2_MAX_PREV_ROWS;
                        preVal = (int32_t)s_nearRows[nearSlot][idx];
                    } 
                    else {
                        size_t farSlot = preRowBase + (size_t)(m % PHASE2_MAX_FAR_ROWS);
                        size_t addr = farSlot * (size_t)PHASE2_MAX_BAND_CELLS + (size_t)idx;
                        preVal = (int32_t)fartherRowValues_d[addr];
                    }
                    int32_t delVal = preVal - del_penalty;
                    if (delVal > localBest) localBest = delVal;
                }
                if (j - 1 >= preStart && j - 1 <= preEnd) {
                    int32_t idxm1 = (j - 1) - preStart;
                    int32_t preValm1;
                    if (useNear) {
                        int32_t nearSlot = m % PHASE2_MAX_PREV_ROWS;
                        preValm1 = (int32_t)s_nearRows[nearSlot][idxm1];
                    } 
                    else {
                        size_t farSlot = preRowBase + (size_t)(m % PHASE2_MAX_FAR_ROWS);
                        size_t addr = farSlot * (size_t)PHASE2_MAX_BAND_CELLS + (size_t)idxm1;
                        preValm1 = (int32_t)fartherRowValues_d[addr];
                    }
                    int32_t matchVal = preValm1 + matchScore;
                    if (matchVal > localBest) localBest = matchVal;
                }
            }
            if (localBest < 0) localBest = 0;
            s_matchDelete[local] = score_cast_i32<T>(localBest);
        }
        __syncthreads();
        if (tid == 0) {
            int32_t current_row_max = INT32_MIN;
            int32_t current_row_max_index_left = rowStart;
            for (int32_t local = 0; local < rowLen; ++local) {
                int32_t insVal = (local == 0) ? (INT32_MIN / 4)
                                              : ((int32_t)s_curr[local - 1] - ins_penalty);
                int32_t val = (int32_t)s_matchDelete[local];
                if (insVal > val) val = insVal;
                if (val < 0) val = 0;
                s_curr[local] = score_cast_i32<T>(val);
                int32_t col = rowStart + local;
                if (col < readLen && val > current_row_max) {
                    current_row_max = val;
                    current_row_max_index_left = col;
                }
            }
            if (current_row_max >= s_bestScore) {
                s_bestScore = current_row_max;
                s_bestCol = i;
                s_bestRow = current_row_max_index_left;
            }
            size_t rowBase = (size_t)readSlot * (size_t)PHASE2_MAX_FAR_ROWS;
            size_t farSlot = rowBase + (size_t)(i % PHASE2_MAX_FAR_ROWS);
            fartherRowId_d[farSlot] = i;
            fartherRowBandStart_d[farSlot] = rowStart;
            fartherRowBandEnd_d[farSlot] = rowEnd;
            fartherRowMaxLeft_d[farSlot] = current_row_max_index_left;
            for (int32_t local = 0; local < rowLen; ++local) {
                fartherRowValues_d[farSlot * (size_t)PHASE2_MAX_BAND_CELLS + (size_t)local] = s_curr[local];
            }
            int32_t slot = i % PHASE2_MAX_PREV_ROWS;
            s_nearRowId[slot] = i;
            s_nearStart[slot] = rowStart;
            s_nearEnd[slot] = rowEnd;
            s_nearMaxLeft[slot] = current_row_max_index_left;
            for (int32_t local = 0; local < rowLen; ++local) {
                s_nearRows[slot][local] = s_curr[local];
            }
            if (s_bestScore == forward_bestScore) {
                s_done = 1;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        bestScores_d[originalReadIdx] = s_bestScore;
        bestRows_d[originalReadIdx] = s_bestRow;
        bestCols_d[originalReadIdx] = s_bestCol;
    }
}

// ========== CUDA Launcher函数 (Templated) ==========

template <typename T>
void launchAlignKernel_Phase1_Template(
    const char* readSetSOA_d,
    const size_t* readSetSOAPrefixSum_d,
    const size_t* sortedReadOrder_d,
    const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d,
    const int32_t* graphAdjcnyIn_d,
    const int32_t* graphOffsetsIn_d,
    const bool* withLongHop_d,
    int32_t* bestScores_d,
    int32_t* bestCols_d,
    int32_t* bestRows_d,
    int32_t match_score,
    int32_t mismatch_penalty,
    int32_t del_penalty,
    int32_t ins_penalty,
    size_t readCount,
    int32_t numVertices)
{
    const int threadsPerBlock = (sizeof(T) == 1) ? 128 : 64;
    const size_t perReadBufferBytes = (size_t)numVertices * 2ULL * sizeof(T);
    if (perReadBufferBytes == 0)
    {
        fprintf(stderr, "ERROR: invalid Phase1 buffer size (numVertices=%d)\n", numVertices);
        return;
    }

    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaError_t memInfoErr = cudaMemGetInfo(&freeMem, &totalMem);
    if (memInfoErr != cudaSuccess)
    {
        fprintf(stderr, "WARN: cudaMemGetInfo failed in Phase1 launcher: %s\n", cudaGetErrorString(memInfoErr));
        freeMem = 0;
    }

    size_t maxChunkReads = readCount;
    if (freeMem > 0)
    {
        const size_t budgetBytes = freeMem / 2;
        size_t byMem = budgetBytes / perReadBufferBytes;
        if (byMem == 0) byMem = 1;
        if (byMem < maxChunkReads) maxChunkReads = byMem;
    }
    if (maxChunkReads == 0) maxChunkReads = 1;

    printf("Launching Phase 1: reads=%lu, chunk=%lu, per-read-buffer=%.2f MB, Type: %s\n",
           (unsigned long)readCount,
           (unsigned long)maxChunkReads,
           (double)perReadBufferBytes / (1024.0 * 1024.0),
           (sizeof(T) == 1) ? "int8_t" : "int16_t");

    T* globalLastBatchRow_d = nullptr;
    size_t allocatedChunkReads = 0;
    cudaError_t allocErr = cudaSuccess;
    while (allocatedChunkReads < maxChunkReads)
    {
        size_t tryReads = maxChunkReads - allocatedChunkReads;
        size_t tryBytes = tryReads * perReadBufferBytes;
        allocErr = cudaMalloc(&globalLastBatchRow_d, tryBytes);
        if (allocErr == cudaSuccess)
        {
            allocatedChunkReads = tryReads;
            break;
        }
        allocatedChunkReads += (maxChunkReads - allocatedChunkReads) / 2 + 1;
    }

    if (globalLastBatchRow_d == nullptr)
    {
        fprintf(stderr, "ERROR: Phase1 failed to allocate temporary DP buffer: %s\n", cudaGetErrorString(allocErr));
        return;
    }

    const size_t sharedMemSize = 0;
    for (size_t chunkStart = 0; chunkStart < readCount; chunkStart += allocatedChunkReads)
    {
        size_t chunkReads = std::min(allocatedChunkReads, readCount - chunkStart);
        size_t chunkBytes = chunkReads * perReadBufferBytes;

        cudaError_t memsetErr = cudaMemset(globalLastBatchRow_d, 0, chunkBytes);
        if (memsetErr != cudaSuccess)
        {
            fprintf(stderr, "ERROR: Phase1 cudaMemset failed at chunk %lu: %s\n",
                    (unsigned long)chunkStart, cudaGetErrorString(memsetErr));
            cudaFree(globalLastBatchRow_d);
            return;
        }

        int blocksPerGrid = (int)((chunkReads + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
        alignKernel_Phase1_GPU_Template<T><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            readSetSOA_d,
            readSetSOAPrefixSum_d + chunkStart,
            sortedReadOrder_d + chunkStart,
            sortedReadLengths_d + chunkStart,
            graphVertexLabel_d,
            graphAdjcnyIn_d,
            graphOffsetsIn_d,
            withLongHop_d,
            bestScores_d,
            bestCols_d,
            bestRows_d,
            match_score,
            mismatch_penalty,
            del_penalty,
            ins_penalty,
            chunkReads,
            numVertices,
            globalLastBatchRow_d);

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess)
        {
            fprintf(stderr, "ERROR: Phase1 kernel launch failed at chunk %lu: %s\n",
                    (unsigned long)chunkStart, cudaGetErrorString(launchErr));
            cudaFree(globalLastBatchRow_d);
            return;
        }

        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess)
        {
            fprintf(stderr, "ERROR: Phase1 kernel sync failed at chunk %lu: %s\n",
                    (unsigned long)chunkStart, cudaGetErrorString(syncErr));
            cudaFree(globalLastBatchRow_d);
            return;
        }
    }

    cudaFree(globalLastBatchRow_d);
}

template <typename T>
void launchAlignKernel_Phase2_Template(
    const char* readSetSOA_d,
    const size_t* readSetSOAPrefixSum_d,
    const size_t* sortedReadOrder_d,
    const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d,
    const int32_t* graphAdjcnyOut_d,
    const int32_t* graphOffsetsOut_d,
    const int32_t* phase1_bestCols_d,
    const int32_t* phase1_bestRows_d,
    const int32_t* phase1_bestScores_d,
    int32_t* bestScores_d,
    int32_t* bestCols_d,
    int32_t* bestRows_d,
    int32_t match_score,
    int32_t mismatch_penalty,
    int32_t del_penalty,
    int32_t ins_penalty,
    size_t readCount,
    int32_t numVertices)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)readCount;
    
    int32_t* fartherRowId_d;
    int32_t* fartherRowBandStart_d;
    int32_t* fartherRowBandEnd_d;
    int32_t* fartherRowMaxLeft_d;
    T* fartherRowValues_d;

    size_t rowCount = readCount * (size_t)PHASE2_MAX_FAR_ROWS;
    size_t rowValuesCount = rowCount * (size_t)PHASE2_MAX_BAND_CELLS;
    cudaMalloc(&fartherRowId_d, rowCount * sizeof(int32_t));
    cudaMalloc(&fartherRowBandStart_d, rowCount * sizeof(int32_t));
    cudaMalloc(&fartherRowBandEnd_d, rowCount * sizeof(int32_t));
    cudaMalloc(&fartherRowMaxLeft_d, rowCount * sizeof(int32_t));
    cudaMalloc(&fartherRowValues_d, rowValuesCount * sizeof(T));

    cudaMemset(fartherRowId_d, 0xFF, rowCount * sizeof(int32_t));
    cudaMemset(fartherRowBandStart_d, 0, rowCount * sizeof(int32_t));
    cudaMemset(fartherRowBandEnd_d, 0xFF, rowCount * sizeof(int32_t));
    cudaMemset(fartherRowMaxLeft_d, 0, rowCount * sizeof(int32_t));
    cudaMemset(fartherRowValues_d, 0, rowValuesCount * sizeof(T));

    size_t sharedMemSize = 0;
    
    printf("Launching Phase 2: %d blocks x %d threads, SharedMem: %lu KB, Type: %s\n", 
           blocksPerGrid, threadsPerBlock, sharedMemSize / 1024, (sizeof(T) == 1) ? "int8_t" : "int16_t");

    // === OPTIMIZATION: Use banded version for better performance ===
    alignKernel_Phase2_GPU_Template_Banded<T><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        readSetSOA_d, readSetSOAPrefixSum_d, sortedReadOrder_d, sortedReadLengths_d,
        graphVertexLabel_d, graphAdjcnyOut_d, graphOffsetsOut_d,
        phase1_bestCols_d, phase1_bestRows_d, phase1_bestScores_d,
        bestScores_d, bestCols_d, bestRows_d,
        match_score, mismatch_penalty, del_penalty, ins_penalty,
        readCount, numVertices,
        fartherRowId_d,
        fartherRowBandStart_d, fartherRowBandEnd_d, fartherRowMaxLeft_d, fartherRowValues_d
    );
    
    cudaDeviceSynchronize();
    cudaFree(fartherRowId_d);
    cudaFree(fartherRowBandStart_d);
    cudaFree(fartherRowBandEnd_d);
    cudaFree(fartherRowMaxLeft_d);
    cudaFree(fartherRowValues_d);
}

// ========== C 接口 (供 C++ 调用) ==========

extern "C" void launchAlignKernel_Phase1_int8(
    const char* readSetSOA_d, const size_t* readSetSOAPrefixSum_d, const size_t* sortedReadOrder_d, const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d, const int32_t* graphAdjcnyIn_d, const int32_t* graphOffsetsIn_d, const bool* withLongHop_d,
    int32_t* bestScores_d, int32_t* bestCols_d, int32_t* bestRows_d,
    int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
    size_t readCount, int32_t numVertices)
{
    launchAlignKernel_Phase1_Template<int8_t>(
        readSetSOA_d, readSetSOAPrefixSum_d, sortedReadOrder_d, sortedReadLengths_d,
        graphVertexLabel_d, graphAdjcnyIn_d, graphOffsetsIn_d, withLongHop_d,
        bestScores_d, bestCols_d, bestRows_d,
        match_score, mismatch_penalty, del_penalty, ins_penalty,
        readCount, numVertices
    );
}

extern "C" void launchAlignKernel_Phase1_int16(
    const char* readSetSOA_d, const size_t* readSetSOAPrefixSum_d, const size_t* sortedReadOrder_d, const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d, const int32_t* graphAdjcnyIn_d, const int32_t* graphOffsetsIn_d, const bool* withLongHop_d,
    int32_t* bestScores_d, int32_t* bestCols_d, int32_t* bestRows_d,
    int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
    size_t readCount, int32_t numVertices)
{
    launchAlignKernel_Phase1_Template<int16_t>(
        readSetSOA_d, readSetSOAPrefixSum_d, sortedReadOrder_d, sortedReadLengths_d,
        graphVertexLabel_d, graphAdjcnyIn_d, graphOffsetsIn_d, withLongHop_d,
        bestScores_d, bestCols_d, bestRows_d,
        match_score, mismatch_penalty, del_penalty, ins_penalty,
        readCount, numVertices
    );
}

extern "C" void launchAlignKernel_Phase2_int8(
    const char* readSetSOA_d, const size_t* readSetSOAPrefixSum_d, const size_t* sortedReadOrder_d, const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d, const int32_t* graphAdjcnyOut_d, const int32_t* graphOffsetsOut_d,
    const int32_t* phase1_bestCols_d, const int32_t* phase1_bestRows_d, const int32_t* phase1_bestScores_d,
    int32_t* bestScores_d, int32_t* bestCols_d, int32_t* bestRows_d,
    int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
    size_t readCount, int32_t numVertices)
{
    launchAlignKernel_Phase2_Template<int8_t>(
        readSetSOA_d, readSetSOAPrefixSum_d, sortedReadOrder_d, sortedReadLengths_d,
        graphVertexLabel_d, graphAdjcnyOut_d, graphOffsetsOut_d,
        phase1_bestCols_d, phase1_bestRows_d, phase1_bestScores_d,
        bestScores_d, bestCols_d, bestRows_d,
        match_score, mismatch_penalty, del_penalty, ins_penalty,
        readCount, numVertices
    );
}

extern "C" void launchAlignKernel_Phase2_int16(
    const char* readSetSOA_d, const size_t* readSetSOAPrefixSum_d, const size_t* sortedReadOrder_d, const size_t* sortedReadLengths_d,
    const char* graphVertexLabel_d, const int32_t* graphAdjcnyOut_d, const int32_t* graphOffsetsOut_d,
    const int32_t* phase1_bestCols_d, const int32_t* phase1_bestRows_d, const int32_t* phase1_bestScores_d,
    int32_t* bestScores_d, int32_t* bestCols_d, int32_t* bestRows_d,
    int32_t match_score, int32_t mismatch_penalty, int32_t del_penalty, int32_t ins_penalty,
    size_t readCount, int32_t numVertices)
{
    launchAlignKernel_Phase2_Template<int16_t>(
        readSetSOA_d, readSetSOAPrefixSum_d, sortedReadOrder_d, sortedReadLengths_d,
        graphVertexLabel_d, graphAdjcnyOut_d, graphOffsetsOut_d,
        phase1_bestCols_d, phase1_bestRows_d, phase1_bestScores_d,
        bestScores_d, bestCols_d, bestRows_d,
        match_score, mismatch_penalty, del_penalty, ins_penalty,
        readCount, numVertices
    );
}
