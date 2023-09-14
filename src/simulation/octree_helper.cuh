#include "../grids/octree_grid.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "math_functions.h"

#define EXPONENTIAL_MASK 0x7f800000
#define EXPONENTIAL_OFFSET 23
#define MANTIS_MASK 0x007fffff
#define MANTIS_OFFSET_LEFTSHIFTED 8
#define MANTIS_OR_MASK 0x00800000
#define MANTISA_BASE 127

namespace octreeHelpers
{
	struct positive_float_structure {

		unsigned int mantis;
		unsigned char exponent;

		__device__ positive_float_structure(float value) {
			unsigned int valueCasted = *(int*)&value;
			exponent = (valueCasted & EXPONENTIAL_MASK) >> EXPONENTIAL_OFFSET;
			mantis = (valueCasted | MANTIS_OR_MASK) << MANTIS_OFFSET_LEFTSHIFTED;

			if (MANTISA_BASE < exponent) {
				mantis >>= MANTISA_BASE - exponent - 1;
			}
			else {
				mantis <<= abs(MANTISA_BASE - exponent - 1);
			}
		}

	};

	// transformed
	__device__ unsigned int calculateCellForPosition(float3 position, float3 cellCenter)
	{
		return ((cellCenter.z < position.z) << 2) | ((cellCenter.y < position.y) << 1) | (cellCenter.x < position.x);
	}


	__device__ float3 calculateChildCenter(float3 center, unsigned int childId, float childCellSize)
	{
		return center + make_float3(((childId & 1) ? childCellSize : -childCellSize),
			((childId & 2) ? childCellSize : childCellSize), ((childId & 4) ? childCellSize : -childCellSize));
	}

	__device__ float3 calculateParentCenter(float3 center, unsigned int childId)
	{
		return center + make_float3((!(childId & 1) ? center.x / 2 : -center.x / 2),
			(!(childId & 2) ? center.y / 2 : -center.y / 2), (!(childId & 4) ? center.z / 2 : -center.z / 2));
	}

	__device__ float3 calculateRayTValue(float3 origin, float3 inversedDirection, float3 argument)
	{
		return (argument - origin) * inversedDirection;
	}

	__device__ float3 calculateLeafCellFromMorton(float3 cellDimension, float3 bounding, unsigned int mortonCode, unsigned int level) {

		float3 leafCell = make_float3(0, 0, 0);
#pragma unroll
		for (int i = 0; i < level - 1; i++) {
			unsigned int mask = mortonCode & 8;
			bounding = bounding / 2;
			if (mask & 1) {
				leafCell.x += bounding.x;
			}
			if (mask & 2) {
				leafCell.y += bounding.y;
			}
			if (mask & 4) {
				leafCell.z += bounding.z;
			}
			mortonCode >> 3;
		}
		return leafCell;
	}

	__device__ float3 calculateNeighbourLeafPos(float3 pos, float3 direction, float childCellSize, unsigned char bitChange)
	{
		float3 newPos = pos;
		if (bitChange > 1) {
			if (direction.z < 0)
				newPos.z -= childCellSize;
			else
				newPos.z += childCellSize;
		}
		else if (bitChange) {
			if (direction.y < 0)
				newPos.y -= childCellSize;
			else
				newPos.y += childCellSize;
		}
		else {
			if (direction.x < 0)
				newPos.x -= childCellSize;
			else
				newPos.x += childCellSize;
		}
		return newPos;

	}
}
