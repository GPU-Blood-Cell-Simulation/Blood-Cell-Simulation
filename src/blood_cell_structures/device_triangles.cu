#include "device_triangles.cuh"

#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>


__global__ void calculateCenters(cudaVec3 position, unsigned int* indices, cudaVec3 centers, int triangleCount)
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

DeviceTriangles::DeviceTriangles(const Mesh& mesh) : triangleCount(mesh.indices.size() / 3), vertexCount(mesh.vertices.size()),
	centers(triangleCount), position(vertexCount), velocity(vertexCount), force(vertexCount), tempForceBuffer(vertexCount),
	threadsPerBlock(vertexCount > 1024 ? 1024 : vertexCount), blDim(std::ceil(float(vertexCount) / threadsPerBlock))
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
	int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	int blocks = (triangleCount + threadsPerBlock - 1) / threadsPerBlock;
	calculateCenters << <blocks, threadsPerBlock >> > (position, indices, centers, triangleCount);
	cudaDeviceSynchronize();
}

DeviceTriangles::DeviceTriangles(const DeviceTriangles& other) : isCopy(true), triangleCount(other.triangleCount), vertexCount(other.vertexCount),
	position(other.position), velocity(other.velocity), force(other.force), indices(other.indices), centers(other.centers), tempForceBuffer(other.tempForceBuffer),
	blDim(other.blDim), threadsPerBlock(other.threadsPerBlock) {}

DeviceTriangles::~DeviceTriangles()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(indices));
	}
}

__global__ void propagateForcesIntoPositionsKernel(DeviceTriangles triangles)
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
void DeviceTriangles::propagateForcesIntoPositions()
{
	propagateForcesIntoPositionsKernel << <blDim, threadsPerBlock >> > (*this);
}


/// <summary>
/// Update the tempForceBuffer based on forces applied onto 4 neighboring vertices in 2D space uisng elastic springs
/// </summary>
/// <param name="force">Vertex force vector</param>
/// <param name="tempForceBuffer">Temporary buffer necessary to synchronize</param>
/// <returns></returns>
__global__ void gatherForcesKernel(DeviceTriangles triangles)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= triangles.force.size)
		return;

	// TODO: vertex distances (spring lengths) are hardcoded for now, ideally we'd like to calculate them for every possible vein model
	constexpr float veinVertexHorizontalDistance = 20.9057f;
	constexpr float veinVertexNonHorizontalDistances[] = { 21.5597f, 5.26999f, 21.5597f };

	float springForce;
	float3 neighborPosition;

	float3 vertexPosition = triangles.position.get(vertex);
	float3 vertexVelocity = triangles.position.get(vertex);
	float3 vertexForce = { 0,0,0 };

	// Calculate our own spatial indices
	unsigned int i = vertex / horizontalLayers;
	unsigned int j = vertex - i * horizontalLayers;

	// vertically adjacent vertices

	unsigned int jSpan[] =
	{
		j != 0 ? j - 1 : horizontalLayers - 1,
		j,
		(j + 1) % horizontalLayers
	};


	unsigned int vertexHorizontalPrev = i * horizontalLayers + jSpan[0];
	unsigned int vertexHorizontalNext = i * horizontalLayers + jSpan[2];


	// Previous horizontally
	neighborPosition = triangles.position.get(vertexHorizontalPrev);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexHorizontalPrev), veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);

	// Next horizontally
	neighborPosition = triangles.position.get(vertexHorizontalNext);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexHorizontalNext), veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);

	

	// not the lower end of the vein
	if (i != 0)
	{
		// Lower vertical neighbors
		#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			unsigned int vertexVerticalPrev = (i - 1) * horizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.position.get(vertexVerticalPrev);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexVerticalPrev), veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}
		
	}

	//// not the upper end of the vein
	if (i != verticalLayers - 1)
	{
		// Upper vertical neighbors
		#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			unsigned int vertexVerticalNext = (i + 1) * horizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.position.get(vertexVerticalNext);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocity.get(vertexVerticalNext), veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}
		
	}

	triangles.tempForceBuffer.set(vertex, vertexForce);
}

__global__ void updateForcesKernel(cudaVec3 force, const cudaVec3 tempForceBuffer)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= force.size)
		return;

	force.add(vertex, tempForceBuffer.get(vertex));
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void DeviceTriangles::gatherForcesFromNeighbors()
{
	gatherForcesKernel << <blDim, threadsPerBlock >> > (*this);

	// Global synchronize - unfortunately necessary as neighboring vertices are not limited to blocks

	updateForcesKernel << <blDim, threadsPerBlock >> > (force, tempForceBuffer);
}
