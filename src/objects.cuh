#include <vector>
#include "defines.cuh"

struct cudaVec3
{
	float* x;
	float* y;
	float* z;
	int size;

	__device__ __host__ cudaVec3() {}
	__device__ __host__ void createVec(int n);

	__device__ __host__ float3 get(int index);
	__device__ __host__ void set(int index, float3 v);
	__device__ __host__ void add(int index, float3 v);
};

// Global structure of particles
struct particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;
	particles() {}

} globalParticles;

// 
struct spring_graph
{
	int par_cnt;
	std::vector<float>* lengths;
	int spring_cntr = 0;

	float* cpu_particle_map;
	int* cpu_shifts;

	float* gpu_particle_map;
	int* gpu_shifts;

	spring_graph() {}
	spring_graph(int n);

	__host__ __device__ void set_spring(int a, int b, float l);
	__host__ __device__ void flush();

	__device__ float* getSpringPtr(int index);
};


class corpuscle
{
public:
	unsigned int* particles_indices;
	unsigned int particles_count;

	spring_graph spr_map;

	float3 center;
	float radius;

	corpuscle(int n);

	virtual __device__ void createCorpuscle(int i, float3 center, particles& p, int p_cnt) = 0;
};

class dipol : public corpuscle
{
public:
	dipol() : corpuscle(2) {}

	virtual __device__ void createCorpuscle(int i, float3 center, particles& p, int p_cnt) override;
};


// global corpuscle array
dipol* corps;
