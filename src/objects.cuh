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

	// to use only on device
	__device__ float3 get(int index);
	__device__ void set(int index, float3 v);
	__device__ void add(int index, float3 v);
};
#define CUDA_VEC_3
#endif


// Global structure of particles
#ifndef PARTICLS
struct particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;
	particles(int n) {
		position.createVec(n);
		velocity.createVec(n);
		force.createVec(n);
	}

};
#define PARTICLS
#endif

#ifndef CORPSCLS
class corpuscles
{
public:
	virtual __device__ void propagateForces(particles& gp, int particleInd) {}
	virtual __device__ void setCorpuscle(int index, float3 center, particles& particls, int p_cnt) {}
};
#define CORPSCLS
#endif

#ifndef DIPLS
class dipols : public corpuscles
{
	float L0;

public:
	dipols(int n, int initialLength = 0.5f){
		L0 = initialLength;
	}

	virtual __device__ void propagateForces(particles& gp, int particleInd) override;
	virtual __device__ void setCorpuscle(int index, float3 center, particles& particls, int p_cnt) override;
};
#define DIPLS
#endif