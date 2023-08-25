#include "vein_triangles.cuh"

#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>



__global__ void calculateCentersKernel(cudaVec3 position, unsigned int* indices, cudaVec3 centers, unsigned int triangleCount)

{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= triangleCount)
		return;
	float3 vv1 = position.get(indices[3 * id]);
	float3 vv2 = position.get(indices[3 * id + 1]);
	float3 vv3 = position.get(indices[3 * id + 2]);

	float x = (vv1.x + vv2.x + vv3.x) / 3;
	float y = (vv1.y + vv2.y + vv3.y) / 3;
	float z = (vv1.z + vv2.z + vv3.z) / 3;
	centers.set(id, make_float3(x, y, z));
}

void VeinTriangles::calculateCenters(unsigned int blocks, unsigned int threadsPerBlock)
{
	calculateCentersKernel << <blocks, threadsPerBlock >> > (position, indices, centers, triangleCount);
}

VeinTriangles::VeinTriangles(const Mesh& mesh, const std::tuple<float, float, float>& springLengths) :
	triangleCount(mesh.indices.size() / 3), vertexCount(mesh.vertices.size()),
	centers(triangleCount), position(vertexCount), velocity(vertexCount), force(vertexCount),
	veinVertexHorizontalDistance(std::get<0>(springLengths)),
	veinVertexNonHorizontalDistances{ std::get<2>(springLengths), std::get<1>(springLengths), std::get<2>(springLengths) }
{
	// allocate
	HANDLE_ERROR(cudaMalloc((void**)&indices, 3 * triangleCount * sizeof(int)));

	std::vector<unsigned int> indicesMem = mesh.indices;

	std::vector<float> vx(vertexCount);
	std::vector<float> vy(vertexCount);
	std::vector<float> vz(vertexCount);

	int iter = 0;
	std::for_each(mesh.vertices.begin(), mesh.vertices.end(), [&](auto& v)
		{
		vx[iter] = v.Position.x;
		vy[iter] = v.Position.y;
		vz[iter++] = v.Position.z;
		});

	// copy
	HANDLE_ERROR(cudaMemcpy(indices, indicesMem.data(), 3 * triangleCount * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(position.x, vx.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(position.y, vy.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(position.z, vz.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));

	// centers
	unsigned int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	unsigned int blocks = std::ceil(static_cast<float>(triangleCount) / threadsPerBlock);
	calculateCenters(triangleCount > 1024 ? 1024 : triangleCount, std::ceil(static_cast<float>(triangleCount) / threadsPerBlock));
}

VeinTriangles::VeinTriangles(const VeinTriangles& other) : isCopy(true), triangleCount(other.triangleCount), vertexCount(other.vertexCount),
	position(other.position), velocity(other.velocity), force(other.force), indices(other.indices), centers(other.centers),
	veinVertexHorizontalDistance(other.veinVertexHorizontalDistance),
	veinVertexNonHorizontalDistances{other.veinVertexNonHorizontalDistances[0], other.veinVertexNonHorizontalDistances[1], other.veinVertexNonHorizontalDistances[2] }
{}

VeinTriangles::~VeinTriangles()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(indices));
	}
}

__global__ void propagateForcesIntoPositionsKernel(VeinTriangles triangles)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;

	if (vertex >= triangles.vertexCount)
		return;

	// propagate forces into velocities
	triangles.velocity.add(vertex, dt * triangles.force.get(vertex));

	// propagate velocities into positions
	triangles.position.add(vertex, dt * triangles.velocity.get(vertex));

	// zero forces
	triangles.force.set(vertex, make_float3(0, 0, 0));
}

/// <summary>
/// Propagate forces -> velocities and velocities->positions
/// </summary>
void VeinTriangles::propagateForcesIntoPositions(unsigned int blocks, unsigned int threadsPerBlock)
{
	propagateForcesIntoPositionsKernel << <blocks, threadsPerBlock >> > (*this);
}


/// <summary>
/// Update the tempForceBuffer based on forces applied onto 4 neighboring vertices in 2D space uisng elastic springs
/// </summary>
/// <param name="force">Vertex force vector</param>horizontalLayers
/// <param name="tempForceBuffer">Temporary buffer necessary to synchronize</param>
/// <returns></returns>
__global__ static void gatherForcesKernel(VeinTriangles triangles)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= triangles.vertexCount)
		return;

	float springForce;
	float3 neighborPosition;

	float3 vertexPosition = triangles.position.get(vertex);
	float3 vertexVelocity = triangles.position.get(vertex);
	float3 vertexForce = { 0,0,0 };

	// Calculate our own spatial indices
	unsigned int i = vertex / cylinderHorizontalLayers;
	unsigned int j = vertex - i * cylinderHorizontalLayers;

	// vertically adjacent vertices

	unsigned int jSpan[] =
	{
		j != 0 ? j - 1 : cylinderHorizontalLayers - 1,
		j,
		(j + 1) % cylinderHorizontalLayers
	};


	unsigned int vertexHorizontalPrev = i * cylinderHorizontalLayers + jSpan[0];
	unsigned int vertexHorizontalNext = i * cylinderHorizontalLayers + jSpan[2];


	// Previous horizontally
	neighborPosition = triangles.position.get(vertexHorizontalPrev);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexHorizontalPrev),
		triangles.veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);

	// Next horizontally
	neighborPosition = triangles.position.get(vertexHorizontalNext);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexHorizontalNext),
		triangles.veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);

	

	// not the lower end of the vein
	if (i != 0)
	{
		// Lower vertical neighbors
		#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			unsigned int vertexVerticalPrev = (i - 1) * cylinderHorizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.position.get(vertexVerticalPrev);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexVerticalPrev),
				triangles.veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}
		
	}

	//// not the upper end of the vein
	if (i != cylinderVerticalLayers - 1)
	{
		// Upper vertical neighbors
		#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			unsigned int vertexVerticalNext = (i + 1) * cylinderHorizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.position.get(vertexVerticalNext);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexVerticalNext),
				triangles.veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}
		
	}

	triangles.force.add(vertex, vertexForce);
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void VeinTriangles::gatherForcesFromNeighbors(unsigned int blocks, unsigned int threadsPerBlock)
{
	gatherForcesKernel << <blocks, threadsPerBlock >> > (*this);
}
