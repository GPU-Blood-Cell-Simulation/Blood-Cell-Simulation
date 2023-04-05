#include "glcontroller.cuh"

#ifdef _WIN32

#include <windows.h>

#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glad/glad.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"


namespace graphics
{
	__global__ void calculateOffsetsKernel(float* devVBO, float* positionX, float* positionY, float* positionZ, unsigned int particleCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;
		devVBO[9 * id + 6] = positionX[id];
		devVBO[9 * id + 7] = positionY[id];
		devVBO[9 * id + 8] = positionZ[id];
	}

	graphics::GLController::GLController() : particleModel("Models/Sphere.obj"), solidColorShader(std::make_shared<Shader>(SolidColorShader()))
	{
		// register OpenGL buffer in CUDA
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&VBOresource, particleModel.getVBO(), cudaGraphicsRegisterFlagsNone));
	}

	void graphics::GLController::calculateOffsets(float* positionX, float* positionY, float* positionZ, unsigned int particleCount)
	{
		// get CUDA a pointer to openGL buffer
		float* devVBO = 0;
		size_t numBytes;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &VBOresource, 0));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devVBO, &numBytes, VBOresource));
		
		// translate our CUDA positions into Vertex offsets
		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
		calculateOffsetsKernel << <blocks, threadsPerBlock >> > (devVBO, positionX, positionY, positionZ, particleCount);

		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &VBOresource, 0));
	}

	void graphics::GLController::draw()
	{
		solidColorShader->use();
		solidColorShader->setMatrix("view", view);
		solidColorShader->setMatrix("projection", projection);
		particleModel.draw(solidColorShader);
	}
}