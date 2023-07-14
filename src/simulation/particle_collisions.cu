#include "particle_collisions.cuh"
#include "../utilities/math.cuh"


namespace sim
{
	// Calculate collisions between particles
	__global__ void detectParticleCollisions(BloodCells cells, unsigned int* gridCellIds, unsigned int* particleIds,
		unsigned int* gridCellStarts, unsigned int* gridCellEnds)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= cells.particleCount)
			return;

		int particleId = particleIds[id];
		float3 p1 = cells.particles.position.get(particleId);


		// Naive implementation
		/*for (int i = 0; i < particleCount; i++)
		{
			if (id == i || i == secondParticle)
				continue;

			float3 p2 = particles.position.get(i);
			if (length(p1 - p2) <= 5.0f)
			{
				particles.force.set(id, 50.0f * normalize(p1 - p2));
			}
		}*/

		// Using uniform grid

		int cellId = gridCellIds[id];

		for (int i = gridCellStarts[cellId]; i <= gridCellEnds[cellId]; i++)
		{
			int secondParticleId = particleIds[i];
			if (particleId == secondParticleId)
				continue;

			float3 p2 = cells.particles.position.get(secondParticleId);
			if (length(p1 - p2) <= 5.0f)
			{
				// Uncoalesced writes - area for optimization
				cells.particles.force.set(particleId, 50.0f * normalize(p1 - p2));
			}
		}
	}
}