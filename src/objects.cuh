#include <vector>
#include "defines.cuh"

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


// Global structure of particles
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

class corpuscles
{
	corpuscles(int n);

	cudaVec3 centers;
public:
	virtual __device__ void propagateForces(particles& gp, int particleInd) = 0;
	virtual __device__ void setCorpuscle(int index, float3 center, particles& particls, int p_cnt) = 0;
}

class dipols: public corpuscles
{
	float L0;
public:
	dipol(int n, int initialLength): corpuscles(n) {
		L0 = initialLength;
	}

	virtual __device__ void propagateForces(particles& gp, int particleInd) override;
	virtual __device__ void setCorpuscle(int index, float3 center, particles& particls, int p_cnt) override;
}