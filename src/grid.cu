#include "grid.cuh"
#include "defines.cuh"

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sim
{
    __global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
        unsigned int* CellIds, unsigned int* ParticleIds, const unsigned int particleCount);

    __global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
        const unsigned int* cellIds, const unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);

    void createUniformGrid(const float* positionX, const float* positionY, const float* positionZ,
        unsigned int* cellIds, unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
    {
        // Calculate launch parameters

        const int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
        const int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

        // 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
        calculateCellIdKernel <<<blocks, threadsPerBlock >>>
            (positionX, positionY, positionZ, cellIds, particleIds, particleCount);

        // 2. Sort particle ids by cell id

        thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(cellIds);
        thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

        thrust::stable_sort_by_key(keys, keys + particleCount, values);

        // 3. Find the start and end of every cell

        calculateStartAndEndOfCellKernel <<<blocks, threadsPerBlock >>>
            (positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);
    }

    __global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
        unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount)
    {
        unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleId >= particleCount)
            return;

        unsigned int cellId =
            static_cast<unsigned int>(positionX[particleId] / cellWidth) +
            static_cast<unsigned int>(positionY[particleId] / cellHeight) +
            static_cast<unsigned int>(positionZ[particleId] / cellDepth);

        particleIds[particleId] = particleId;
        cellIds[particleId] = cellId;

    }

    __global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
        const unsigned int* cellIds, const unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
    {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= particleCount)
            return;

        unsigned int currentCellId = cellIds[id];

        // Check if the previous cell id was different - it would mean we found the start of a cell
        if (id > 0 && currentCellId != cellIds[id - 1])
        {
            cellStarts[currentCellId] = id;
        }

        // Check if the next cell id was different - it would mean we found the end of a cell
        if (id < particleCount - 1 && currentCellId != cellIds[id + 1])
        {
            cellEnds[currentCellId] = id;
        }

        // debug
        //if (id < 5) printf("%d. particle id: %d, cell: %d\n", id, particleIds[id], cellIds[id]);

    }
}