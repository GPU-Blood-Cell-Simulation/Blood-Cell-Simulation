#include "structs.cuh"
#include "defines.cuh"
#include <cmath>
#include <map>
#include <utility>
#include <vector>


// Global structure of particles
struct particles
{
	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

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

	void set_spring(int a, int b, float l);
	void flush();

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

	virtual __device__ void createCorpuscle(int i, float3 center) = 0;
};

class dipol : public corpuscle
{
public:
	dipol() : corpuscle(2) {}

	virtual __device__ void createCorpuscle(int i, float3 center) override;
};


// global corpuscle array
corpuscle* corps;
