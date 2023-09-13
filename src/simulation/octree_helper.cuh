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

		positive_float_structure(float value) {
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


	/// <summary>
	/// Function to simulate traversal through octree grid
	/// </summary>
	/// <param name="origin">- ray origin defined in world coords</param>
	/// <param name="direction">- ray direction</param>
	/// <param name="end">- ray end defined in world coords</param>
	/// <param name="masks">- octree nodes masks</param>
	/// <param name="treeData">- pointers to leaves</param>
	/// <param name="maxLevel">- octree depth in levels</param>
	/// <returns>if collision occured with collision data reference parameter</returns>
	__device__ void traverseGrid(float3 origin, float3 direction, float3 end, unsigned char* masks, unsigned int* treeData, const unsigned int maxLevel)
	{
		// necessary parameters
		const float3 bounding = make_float3(width, height, depth);
		const uint8_t s_max = maxLevel;
		const unsigned int leafShift = (ldexp((double)1, 3 * (maxLevel - 1)) - 1) / 7; // (8^(maxL - 1) - 1)/7
		
		const float3 normalizedOrigin = origin / bounding;
		const float3 normalizedEnd = end / bounding;
		const float3 normalizedDirection = direction / bounding;

		const float3 directionSigns = make_float3(!(direction.x < 0), !(direction.y < 0), !(direction.z < 0)); // 1 plus or zero, 0 minus
		const float3 inversedDirection = make_float3(1.0f / normalizedDirection.x, 1.0f / normalizedDirection.y, 1.0f / normalizedDirection.z);

		// values
		unsigned int parentId = 0; // root
		float3 pos = make_float3( 0.5f , 0.5f , 0.5f ); // initial pos is center of root
		uint8_t scale = s_max - 1;
		uint8_t childId = calculateCellForPosition(normalizedOrigin, pos);
		unsigned int realChildId = 8 * parentId + childId + 1;
		float childCellSize = .5f; // normalized cell size

		float3 t = make_float3( 0,0,0 );
		float tEndSquare = length_squared(normalizedEnd - normalizedOrigin);

		while (true) {

			// traversing down the tree to the leaf
			if (scale > 1) {
				childCellSize = 0.5f * childCellSize;
				parentId = realChildId;
				pos = calculateChildCenter(pos, childId, childCellSize);
				childId = calculateCellForPosition(normalizedOrigin, pos);
				unsigned int realChildId = 8 * parentId + childId + 1;

				if (!(masks[realChildId] & (1 << childId))) // empty cell
					break;

				scale--;
				continue;
			}

			unsigned int leafMortonCode = treeData[realChildId - leafShift];
			// compute intersections in current cell
			// TODO


			//float3 cellBegining = calculateLeafCellFromMorton(scale, bounding, leafMortonCode, currentLevel);
			float3 cellBeginning = pos - make_float3(fmodf(pos.x, childCellSize), 
				fmod(pos.y, childCellSize), fmod(pos.z, childCellSize));

			// maybe conditions instead of directionSigns ???
			t = calculateRayTValue(normalizedOrigin, inversedDirection, cellBeginning + childCellSize*directionSigns);
			float tMax = vmin(t);

			// break if ray ends before next cell
			if (tMax > tEndSquare)
				break;
			
			bool changeParent = false;
			unsigned char bitChange = 0;

			// bit changing && should be + && is minus
			if (!(tMax > t.x) && (childId & 1) && direction.x < 0) {
				changeParent = true;
			}
			else if (!(tMax > t.x)) {
				bitChange = 1;
				if ((childId & 1) && direction.y < 0) {
					changeParent = true;
				}
			}
			else if (!(tMax > t.x)) {
				bitChange = 2;
				if ((childId & 1) && direction.z < 0) {
					changeParent = true;
				}
			}

			if (changeParent) {
				// calculate new pos
				float3 newPos = calculateNeighbourLeafPos(pos, normalizedDirection, childCellSize, bitChange);

				positive_float_structure posX(pos.x), posY(pos.y), posZ(pos.z);
				positive_float_structure newPosX(newPos.x), newPosY(newPos.y), newPosZ(newPos.z);

				uint8_t minMantisaShift = 31 - log2((double)max(posX.mantis ^ newPosX.mantis,
					max(posY.mantis ^ newPosY.mantis, posZ.mantis ^ newPosZ.mantis)));

				scale = s_max - minMantisaShift - 1;
				childCellSize = ldexp((double)1, scale - s_max);

				realChildId = 0;
				unsigned int shift = 0x80000000;
				uint8_t mask = 0;
				for (int i = 31; i > 31 - minMantisaShift; --i) {
					mask = (newPosX.mantis & shift) |
						((newPosY.mantis & shift) << 1) | ((newPosZ.mantis & shift) << 2);
					realChildId += mask;
					realChildId *= 8;
					shift >>= 1;
				}
				childId = mask; // last mask value;
				parentId = (unsigned int)(realChildId * 0.125f); // divide by 8

			}
			else {

				// calculate new childId
				childId ^= 1 << bitChange;
				unsigned int realChildId = 8 * parentId + childId + 1;

				if (!(masks[realChildId] & (1 << childId))) // empty cell
					break;

				// calculate new pos
				pos = calculateNeighbourLeafPos(pos, normalizedDirection, childCellSize, bitChange);
			}
		}
	}
}
