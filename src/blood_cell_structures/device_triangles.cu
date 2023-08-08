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
	centers(triangleCount), position(vertexCount), velocity(vertexCount), force(vertexCount), tempForcesBuffer(vertexCount),
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
	position(other.position), velocity(other.velocity), force(other.force), indices(other.indices), centers(other.centers), tempForcesBuffer(other.tempForcesBuffer),
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

	//constexpr int verticalLayers = cylinderScaleY * cylinderHeight;
	//constexpr int horizontalLayers = cylinderScaleX * cylinderRadius;

	//int i = vertex / horizontalLayers;
	//int j = vertex - i * horizontalLayers;

	//// lower end of the vein
	//if (j == 0)
	//{

	//}
	//// upper end of the vein
	//else if (j == horizontalLayers - 1)
	//{

	//}
	//else
	//{

	//}

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
/// Update the tempForceBuffer based on forces applied onto 4 neighboring vertices in 2D space
/// </summary>
/// <param name="force">Vertex force vector</param>
/// <param name="tempForceBuffer">Temporary buffer necessary to synchronize</param>
/// <returns></returns>
__global__ void gatherForcesKernel(const cudaVec3 force, cudaVec3 tempForceBuffer)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= force.size)
		return;

	// TODO: gather forces from all neighbors

	tempForceBuffer.set(vertex, force.get(vertex));
}

__global__ void updateForcesKernel(cudaVec3 force, const cudaVec3 tempForceBuffer)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= force.size)
		return;

	force.set(vertex, tempForceBuffer.get(vertex));
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void DeviceTriangles::gatherForcesFromNeighbors()
{
	gatherForcesKernel << <blDim, threadsPerBlock >> > (force, tempForcesBuffer);

	// Global synchronize - unfortunately necessary as neighboring vertices are not limited to blocks

	updateForcesKernel << <blDim, threadsPerBlock >> > (force, tempForcesBuffer);
}
