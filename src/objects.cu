#include "objects.cuh"


/// spring array

spring_graph::spring_graph(int n)
{
	par_cnt = n;
	lengths = new std::vector<float>[n];
}

void spring_graph::set_spring(int a, int b, float l)
{
	lengths[a].push_back(b);
	lengths[a].push_back(l);
	spring_cntr++;
}

void spring_graph::flush()
{
	int totalS = par_cnt + 2 * spring_cntr;

	// copy vectors into cpu array
	cpu_particle_map = new float[totalS];
	cpu_shifts = new int[par_cnt];
	int shift = 0;

	for (int i = 0; i < par_cnt; ++i)
	{
		cpu_shifts[i] = shift;
		int size = lengths[i].size();
		cpu_particle_map[shift++] = (float)size;

		std::copy(lengths[i].begin(), lengths[i].end(), cpu_particle_map + shift);
		shift += size;
	}

	//copy to gpu
	HANDLE_ERROR(cudaMalloc((void**)&gpu_particle_map, totalS * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_shifts, par_cnt * sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(gpu_particle_map, cpu_particle_map, totalS, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gpu_shifts, cpu_shifts, par_cnt, cudaMemcpyHostToDevice));
}

__device__ float* spring_graph::getSpringPtr(int index)
{
	return gpu_particle_map + gpu_shifts[index];
}


// corpuscle

corpuscle::corpuscle(int n)
{
	particles_count = n;
	HANDLE_ERROR(cudaMalloc((void**)&particles_indices, n * sizeof(unsigned int)));
	spr_map = spring_graph(n);
}

// dipol
	
void dipol::createCorpuscle(int i, float3 center, particles& particls, int p_cnt)
{
	if (2 * i < p_cnt)
	{
		this->center = center;
		radius = 0.25f;
		particls.position.set(2 * i,		make_float3(0, 0, -0.25f) + center);
		particls.position.set(2 * i + 1, make_float3(0, 0,  0.25f) + center);

		particls.velocity.set(2 * i,		make_float3(0, 0, v0));
		particls.velocity.set(2 * i + 1, make_float3(0, 0, v0));

		particls.force.set(2 * i,	 make_float3(0, 0, 0));
		particls.force.set(2 * i + 1, make_float3(0, 0, 0));

		particles_count = 2;
		spr_map.set_spring(0, 1, 0.5f);
		spr_map.flush();
	}
}