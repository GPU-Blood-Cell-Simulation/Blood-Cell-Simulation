#ifndef VEIN_COLLISIONS_H
#define VEIN_COLLISIONS_H

#include "../utilities/math.cuh"
#include "../blood_cell_structures/blood_cells.cuh"
#include "../blood_cell_structures/device_triangles.cuh"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace sim
{
	struct ray
	{
		float3 origin;
		float3 direction;
		float t = 1e10f;

		// rays may be used to determine intersection with objects
		// so its easy to store object index inside ray
		unsigned int objectIndex = -1;

		__device__ ray(float3 origin, float3 direction);
	};

	__global__ void detectVeinCollisionsAndPropagateParticles(BloodCells cells, DeviceTriangles triangles, int triangleCount);

	__device__ bool calculateSideCollisions(float3 p, ray& r, DeviceTriangles& triangles, int triangleCount);
}

#endif
