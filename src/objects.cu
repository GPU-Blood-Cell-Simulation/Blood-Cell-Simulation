#include "objects.cuh"
#include "defines.cuh"

// cudaVec3

void cudaVec3::createVec(int n)
{
    size = n;
	// allocate 
	HANDLE_ERROR(cudaMalloc((void**)&x, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&y, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&z, n * sizeof(float)));
}

__device__ float3 cudaVec3::get(int index)
{
	return make_float3(x[index], y[index], z[index]);
}

__device__ void cudaVec3::set(int index, float3 v)
{
	x[index] = v.x;
	y[index] = v.y;
	z[index] = v.z;
}

__device__ void cudaVec3::add(int index, float3 v)
{
	x[index] += v.x;
	y[index] += v.y;
	z[index] += v.z;
}

// corpuscles

corpuscles::corpuscles(int n)
{
    centers.createVec3(n);
}

// dipols 

__device__ void dipol::propagateForces(particles& gp, int particleInd)
{
    int secondParticle = particleInd%2 == 0 ? particleInd + 1 : particleInd - 1;
	float3 p1 = gp.position.get(particleInd);
	float3 p2 = gp.position.get(secondParticle);
	float3 v1 = gp.velocity.get(particleInd);
	float3 v2 = gp.velocity.get(secondParticle);
    
	float Fr = (length(p1 - p2) - L0) * k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * d_fact;
    
	gp.force.add(part_index, Fr * normalize(p2 - p1));
	gp.force.add(oth_part, Fr * normalize(p1 - p2));
}

__device__ void dipol::setCorpuscle(int index, float3 center, particles& particls, int p_cnt)
{
	if(2*i < p_cnt)
	{
		centers.set(index, center);
		particls.position.set(2 * i,		make_float3(0, 0, -0.25f) + center);
		particls.position.set(2 * i + 1, make_float3(0, 0,  0.25f) + center);

		particls.velocity.set(2 * i,		make_float3(0, 0, v0));
		particls.velocity.set(2 * i + 1, make_float3(0, 0, v0));

		particls.force.set(2 * i,	 make_float3(0, 0, 0));
		particls.force.set(2 * i + 1, make_float3(0, 0, 0));
	}
}