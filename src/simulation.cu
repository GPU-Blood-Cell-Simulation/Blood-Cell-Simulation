#include "simulation.cuh"
#include "grid.cuh"
#include "defines.cuh"
#include "physics.cuh"
#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sim
{

    __global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

    __global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount);

    __global__ void generateInitialPositionsKernel(particles par, corpuscles* crps, float3 dims, int par_cnt);

    // Allocate GPU buffers for the position vectors
    void allocateMemory(unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
        const unsigned int particleCount)
    {
        HANDLE_ERROR(cudaMalloc((void**)cellIds, particleCount * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)particleIds, particleCount * sizeof(unsigned int)));

        HANDLE_ERROR(cudaMalloc((void**)cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
    }

    // initial position approach
    // arg_0 should be a^2 * arg_1
    // but it is no necessary
    void generateInitialPositionsInLayers(particles& p, corpuscles& c, const int particleCount, const int layersCount)
    {
        int model_par_cnt = 2; /*cell model particles count, 2 for dipol*/
        int corpusclesPerLayer = particleCount / layersCount / model_par_cnt;
        int layerDim = sqrt(corpusclesPerLayer); // assuming layer is a square

        //real_particles_count = layerDim * layerDim * layersCount;

        int threadsPerBlockDim = std::min(layerDim, 32);
        int blDim = std::ceil(float(corpusclesPerLayer) / threadsPerBlockDim);
        dim3 blocks = dim3(blDim, blDim, layersCount);
        dim3 threads = dim3(threadsPerBlockDim, threadsPerBlockDim, 1);

        generateInitialPositionsKernel <<<blocks, threads >>>(p, &c,
            make_float3(width, height, float(layersCount * depth) / 100 )  , particleCount);
    }

    // Generate initial positions and velocities of particles
    void generateRandomPositions(float* positionX, float* positionY, float* positionZ, const int particleCount)
    {
        int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
        int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

        // Set up random seeds
        curandState* devStates;
        cudaMalloc(&devStates, particleCount * sizeof(curandState));
        srand(static_cast<unsigned int>(time(0)));
        int seed = rand();
        setupCurandStatesKernel <<<blocks, threadsPerBlock >>>(devStates, seed, particleCount);

        // Generate random positions and velocity vectors

        generateRandomPositionsKernel <<<blocks, threadsPerBlock >>>(devStates, positionX, positionY, positionZ, particleCount);

        cudaFree(devStates);
    }

    __global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= particleCount)
            return;
        curand_init(seed, id, 0, &states[id]);
    }

    // Generate random positions and velocities at the beginning
    __global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= particleCount)
            return;

        positionX[id] = curand_uniform(&states[id]) * width;
        positionY[id] = curand_uniform(&states[id]) * height;
        positionZ[id] = curand_uniform(&states[id]) * depth;
    }

    

    __global__ void generateInitialPositionsKernel(particles par, corpuscles* crps, float3 dims, int par_cnt)
    {
        int thCnt = blockDim.x * blockDim.y;
        int blCnt2d = gridDim.x * gridDim.y;
        int tid = (blockIdx.z - 1) * blCnt2d * thCnt + (blockIdx.y - 1) * gridDim.x * thCnt 
            + (blockIdx.x - 1) * thCnt + (threadIdx.y - 1) * blockDim.x + threadIdx.x;

        float x = float(((blockIdx.x - 1) * blockDim.x + threadIdx.x) * dims.x) / (blockDim.x * gridDim.x);
        float y = float(((blockIdx.y - 1) * blockDim.y + threadIdx.y) * dims.y) / (blockDim.y * gridDim.y);
        float z = float(((blockIdx.z - 1) * blockDim.z + threadIdx.z) * dims.z * blockDim.z) / (blockDim.z * gridDim.z*100);

        if (x < dims.x && y < dims.y)
        {
            crps->setCorpuscle(tid, make_float3(x, y, z), par, par_cnt);
        }

        // TODO
        /////crps[tid].createCorpuscle(tid, make_float3(x, y, z), par, par_cnt);

        // maybe we should set initial velocity here (or in another kernel)?
        // as for now it is done in createCorpuscle call
        /*for (int i = 0; i < crps[tid].particles_count; ++i)
        {
            par.velocity.set(crps[tid].particles_indices[i], make_float3(0, 0, v0));
        }*/
    }

    void calculateNextFrame(particles& particls, corpuscles& corpuscls, unsigned int* cellIds, unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
    {
        // 1. calculate grid
        createUniformGrid(particls.position.x, particls.position.y, particls.position.z, 
            cellIds, particleIds, cellStarts, cellEnds, particleCount);

        int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
        int blDim = std::ceil(float(particleCount)/ threadsPerBlock);
        // 2. TODO: detections
        physics::propagateParticles << < dim3(blDim), threadsPerBlock >> > (particls, corpuscls);
    }
}
