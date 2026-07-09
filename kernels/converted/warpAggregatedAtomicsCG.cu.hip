/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups/reduce.h>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s at %s:%d\n",                         \
                    hipGetErrorString(err), __FILE__, __LINE__);                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

namespace cg = cooperative_groups;

#define NUM_ELEMS             10000000
#define NUM_THREADS_PER_BLOCK 512

// warp-aggregated atomic increment
__device__ int atomicAggInc(int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();

    // leader does the update
    int res = 0;
    if (active.thread_rank() == 0) {
        res = atomicAdd(counter, active.size());
    }

    // broadcast result
    res = active.shfl(res, 0);

    // each thread computes its own value
    return res + active.thread_rank();
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = id; i < n; i += gridDim.x * blockDim.x) {
        if (src[i] > 0)
            dst[atomicAggInc(nres)] = src[i];
    }
}

// warp-aggregated atomic multi bucket increment
#if defined(__HIP_DEVICE_COMPILE__)
__device__ int atomicAggIncMulti(const int bucket, int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();
    // group all threads with same bucket value.
    auto labeledGroup = cg::labeled_partition(active, bucket);

    int res = 0;
    if (labeledGroup.thread_rank() == 0) {
        res = atomicAdd(&counter[bucket], labeledGroup.size());
    }

    // broadcast result
    res = labeledGroup.shfl(res, 0);

    // each thread computes its own value
    return res + labeledGroup.thread_rank();
}
#endif

// Places individual value indices into its corresponding buckets.
__global__ void
mapToBuckets(const int *srcArr, int *indicesBuckets, int *bucketCounters, const int srcSize, const int numOfBuckets)
{
#if defined(__HIP_DEVICE_COMPILE__)
    cg::grid_group grid = cg::this_grid();

    for (int i = grid.thread_rank(); i < srcSize; i += grid.size()) {
        const int bucket = srcArr[i];
        if (bucket < numOfBuckets) {
            indicesBuckets[atomicAggIncMulti(bucket, bucketCounters)] = i;
        }
    }
#endif
}

int mapIndicesToBuckets(int *h_srcArr, int *d_srcArr, int numOfBuckets)
{
    int *d_indicesBuckets, *d_bucketCounters;
    int *cpuBucketCounters = new int[numOfBuckets];
    int *h_bucketCounters  = new int[numOfBuckets];

    memset(cpuBucketCounters, 0, sizeof(int) * numOfBuckets);
    // Initialize each bucket counters.
    for (int i = 0; i < numOfBuckets; i++) {
        h_bucketCounters[i] = i * NUM_ELEMS;
    }

    HIP_CHECK(hipMalloc(&d_indicesBuckets, sizeof(int) * NUM_ELEMS * numOfBuckets));
    HIP_CHECK(hipMalloc(&d_bucketCounters, sizeof(int) * numOfBuckets));

    HIP_CHECK(hipMemcpy(d_bucketCounters, h_bucketCounters, sizeof(int) * numOfBuckets, hipMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK), 1, 1);

    hipLaunchKernelGGL(mapToBuckets, dimGrid, dimBlock, 0, 0, d_srcArr, d_indicesBuckets, d_bucketCounters, NUM_ELEMS, numOfBuckets);

    HIP_CHECK(hipMemcpy(h_bucketCounters, d_bucketCounters, sizeof(int) * numOfBuckets, hipMemcpyDeviceToHost));

    for (int i = 0; i < NUM_ELEMS; i++) {
        cpuBucketCounters[h_srcArr[i]]++;
    }

    bool allMatch   = true;
    int  finalElems = 0;
    for (int i = 0; i < numOfBuckets; i++) {
        finalElems += (h_bucketCounters[i] - i * NUM_ELEMS);
        if (cpuBucketCounters[i] != (h_bucketCounters[i] - i * NUM_ELEMS)) {
            allMatch = false;
            break;
        }
    }

    if (!allMatch && finalElems != NUM_ELEMS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// Warp-aggregated atomic Max in multi bucket
#if defined(__HIP_DEVICE_COMPILE__)
__device__ void atomicAggMaxMulti(const int bucket, int *counter, const int valueForMax)
{
    cg::coalesced_group active = cg::coalesced_threads();
    // group all threads with same bucket value.
    auto labeledGroup = cg::labeled_partition(active, bucket);

    const int maxValueInGroup = cg::reduce(labeledGroup, valueForMax, cg::greater<int>());

    if (labeledGroup.thread_rank() == 0) {
        atomicMax(&counter[bucket], maxValueInGroup);
    }
}
#endif

// Performs max calculation in each buckets.
__global__ void calculateMaxInEachBuckets(const int *srcArr,
                                          const int *valueInBuckets,
                                          int       *bucketsMax,
                                          const int  srcSize,
                                          const int  numOfBuckets)
{
#if defined(__HIP_DEVICE_COMPILE__)
    cg::grid_group grid = cg::this_grid();

    for (int i = grid.thread_rank(); i < srcSize; i += grid.size()) {
        const int bucket = srcArr[i];
        if (bucket < numOfBuckets) {
            atomicAggMaxMulti(bucket, bucketsMax, valueInBuckets[i]);
        }
    }
#endif
}

int calculateMaxInBuckets(int *h_srcArr, int *d_srcArr, int numOfBuckets)
{
    int *d_valueInBuckets, *d_bucketsMax;
    int *h_valueInBuckets = new int[NUM_ELEMS];
    int *cpuBucketsMax    = new int[numOfBuckets];
    int *h_bucketsMax     = new int[numOfBuckets];

    memset(cpuBucketsMax, 0, sizeof(int) * numOfBuckets);

    // Here we create values which is assumed to correspond to each
    // buckets of srcArr at same array index.
    for (int i = 0; i < NUM_ELEMS; i++) {
        h_valueInBuckets[i] = rand();
    }

    HIP_CHECK(hipMalloc(&d_valueInBuckets, sizeof(int) * NUM_ELEMS));
    HIP_CHECK(hipMalloc(&d_bucketsMax, sizeof(int) * numOfBuckets));

    HIP_CHECK(hipMemset(d_bucketsMax, 0, sizeof(int) * numOfBuckets));
    HIP_CHECK(hipMemcpy(d_valueInBuckets, h_valueInBuckets, sizeof(int) * NUM_ELEMS, hipMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK), 1, 1);

    hipLaunchKernelGGL(calculateMaxInEachBuckets, dimGrid, dimBlock, 0, 0, d_srcArr, d_valueInBuckets, d_bucketsMax, NUM_ELEMS, numOfBuckets);

    HIP_CHECK(hipMemcpy(h_bucketsMax, d_bucketsMax, sizeof(int) * numOfBuckets, hipMemcpyDeviceToHost));

    for (int i = 0; i < NUM_ELEMS; i++) {
        if (cpuBucketsMax[h_srcArr[i]] < h_valueInBuckets[i]) {
            cpuBucketsMax[h_srcArr[i]] = h_valueInBuckets[i];
        }
    }

    bool allMatch   = true;
    int  finalElems = 0;
    for (int i = 0; i < numOfBuckets; i++) {
        if (cpuBucketsMax[i] != h_bucketsMax[i]) {
            allMatch = false;
            printf("CPU i=%d  max = %d mismatches GPU max = %d\n", i, cpuBucketsMax[i], h_bucketsMax[i]);
            break;
        }
    }
    if (allMatch) {
        printf("CPU max matches GPU max\n");
    }

    delete[] h_valueInBuckets;
    delete[] cpuBucketsMax;
    delete[] h_bucketsMax;
    HIP_CHECK(hipFree(d_valueInBuckets));
    HIP_CHECK(hipFree(d_bucketsMax));

    if (!allMatch && finalElems != NUM_ELEMS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    int *data_to_filter, *filtered_data, nres = 0;
    int *d_data_to_filter, *d_filtered_data, *d_nres;

    int numOfBuckets = 5;

    data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

    // Generate input data.
    for (int i = 0; i < NUM_ELEMS; i++) {
        data_to_filter[i] = rand() % numOfBuckets;
    }

    // Simplified device selection: use device 0
    int devId = 0;
    HIP_CHECK(hipSetDevice(devId));

    HIP_CHECK(hipMalloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS));
    HIP_CHECK(hipMalloc(&d_filtered_data, sizeof(int) * NUM_ELEMS));
    HIP_CHECK(hipMalloc(&d_nres, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_data_to_filter, data_to_filter, sizeof(int) * NUM_ELEMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_nres, 0, sizeof(int)));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK) + 1, 1, 1);

    hipLaunchKernelGGL(filter_arr, dimGrid, dimBlock, 0, 0, d_filtered_data, d_nres, d_data_to_filter, NUM_ELEMS);

    HIP_CHECK(hipMemcpy(&nres, d_nres, sizeof(int), hipMemcpyDeviceToHost));

    filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

    HIP_CHECK(hipMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres, hipMemcpyDeviceToHost));

    int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

    // Generate host output with host filtering code.
    int host_flt_count = 0;
    for (int i = 0; i < NUM_ELEMS; i++) {
        if (data_to_filter[i] > 0) {
            host_filtered_data[host_flt_count++] = data_to_filter[i];
        }
    }

    int major = 0;
    HIP_CHECK(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, devId));

    int mapIndicesToBucketsStatus   = EXIT_SUCCESS;
    int calculateMaxInBucketsStatus = EXIT_SUCCESS;
    // atomicAggIncMulti & atomicAggMaxMulti require a GPU of Volta (SM7X) architecture or higher,
    // so that it can take advantage of the new MATCH capability of Volta hardware
    if (major >= 7) {
        mapIndicesToBucketsStatus   = mapIndicesToBuckets(data_to_filter, d_data_to_filter, numOfBuckets);
        calculateMaxInBucketsStatus = calculateMaxInBuckets(data_to_filter, d_data_to_filter, numOfBuckets);
    }

    printf("\nWarp Aggregated Atomics %s \n",
           (host_flt_count == nres) && (mapIndicesToBucketsStatus == EXIT_SUCCESS)
                   && (calculateMaxInBucketsStatus == EXIT_SUCCESS)
               ? "PASSED"
               : "FAILED");

    HIP_CHECK(hipFree(d_data_to_filter));
    HIP_CHECK(hipFree(d_filtered_data));
    HIP_CHECK(hipFree(d_nres));
    free(data_to_filter);
    free(filtered_data);
    free(host_filtered_data);
}