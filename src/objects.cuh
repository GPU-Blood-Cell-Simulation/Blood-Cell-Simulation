#include <vector>
#include "defines.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#ifndef CUDA_VEC_3
struct cudaVec3
{
	float* x;
	float* y;
	float* z;
	int size;

	// allocated on host
	cudaVec3() {}
	void createVec(int n);
	void freeVec();

	// to use only on device
	__device__ float3 get(int index);
	__device__ void set(int index, float3 v);
	__device__ void add(int index, float3 v);
};
#define CUDA_VEC_3
#endif


// Global structure of particles
#ifndef PARTICLS
struct Particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	Particles(int n) {
		position.createVec(n);
		velocity.createVec(n);
		force.createVec(n);
	}

	void freeParticles() {
		position.freeVec();
		velocity.freeVec();
		force.freeVec();
	}

};
#define PARTICLS
#endif

#ifndef CORPSCLS
class corpuscles
{
public:
	virtual __device__ void propagateForces(Particles gp, int particleInd) {}
	virtual __device__ void setCorpuscle(int index, float3 center, Particles particls, int p_cnt) {}
};
#define CORPSCLS
#endif
