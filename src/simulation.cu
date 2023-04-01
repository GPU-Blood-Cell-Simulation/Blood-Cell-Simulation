#include "simulation.cuh"
#include "grid.cuh"
#include "physics.cuh"

#include <ctime>
#include <cmath>

namespace sim
{
    // Allocate GPU buffers
    void allocateMemory(unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
        const unsigned int particleCount)
    {
        globalParticles.position = cudaVec3(particleCount);
        globalParticles.velocity = cudaVec3(particleCount);
        globalParticles.force = cudaVec3(particleCount);

        corps = new dipol[particleCount / 2];

        HANDLE_ERROR(cudaMalloc((void**)cellIds, particleCount * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)particleIds, particleCount * sizeof(unsigned int)));

        HANDLE_ERROR(cudaMalloc((void**)cellStarts, dimension.x / cellWidth * dimension.y / cellHeight * dimension.z / cellDepth * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)cellEnds, dimension.x / cellWidth * dimension.y / cellHeight * dimension.z / cellDepth * sizeof(unsigned int)));
    }

    // initial position approach
    // arg_0 should be a^2 * arg_1
    // but it is no necessary
    void generateInitialPositionsInLayers(const int particleCount, const int layersCount)
    {
        int model_par_cnt = 2; /*cell model particles count*/
        int corpusclesPerLayer = particleCount / layersCount / model_par_cnt;
        int layerDim = sqrt(corpusclesPerLayer); // assuming layer is a square
        real_particles_count = layerDim * layerDim * layersCount;

        int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
        int blDim = std::ceil(layerDim / threadsPerBlock);
        dim3 blocks = dim3(blDim, blDim, layersCount);
        generateInitialPositionsKernel << <blocks, threadsPerBlock >> >
            (globalParticle, corps, make_float3(dimension.x, dimension.y, dimension.z * layers / 100));
    }

    __global__ void generateInitialPositionsKernel(particles& par, corpuscle* crps, float3 dims)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int tid = blockId * blockDim.x + threadIdx.x;

        int w = gridDim.x * blockDim.x;
        int h = gridDim.y * blockDim.y;
        int d = gridDim.z * blockDim.z;

        float x = ((tid % (w*h)) % w) * dims.x/w;
        float y = ((tid % (w*h)) / w) * dims.y/h;
        float z = (tid / (w * h)    ) * dims.z/d;

        crps[tid].createCorpuscle(tid, make_float3(x, y, z));

        // maybe we should set initial velocity here (or in another kernel)?
        // as for now it is done in createCorpuscle call
        /*for (int i = 0; i < crps[tid].particles_count; ++i)
        {
            par.velocity.set(crps[tid].particles_indices[i], make_float3(0, 0, v0));
        }*/
    }

    void calculateNextFrame(float* positionX, float* positionY, float* positionZ,
        unsigned int* cellIds, unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
    {
        // 1. calculate grid
        createUniformGrid(positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);

        // 2. TODO: detections
        physics::propagateParticles << < /*TODO*/ >> > (globalParticles);
    }


    ///// RANDOM POSITIONS


    __global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

    __global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount);

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
        setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, particleCount);

        // Generate random positions and velocity vectors

        generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, positionX, positionY, positionZ, particleCount);

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

        positionX[id] = curand_uniform(&states[id]) * dimension.x;
        positionY[id] = curand_uniform(&states[id]) * dimension.y;
        positionZ[id] = curand_uniform(&states[id]) * dimension.z;
    }
}
