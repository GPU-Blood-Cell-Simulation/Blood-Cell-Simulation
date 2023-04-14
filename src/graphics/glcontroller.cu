#include "glcontroller.cuh"

#ifdef _WIN32

#include <windows.h>

#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glad/glad.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"


namespace graphics
{
	__global__ void calculateOffsetsKernel(float* devCudaOffsetBuffer, float* positionX, float* positionY, float* positionZ, unsigned int particleCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		positionX[id] += id % 2 == 0 ? 0.1 : -0.1;

		devCudaOffsetBuffer[3 *id] = positionX[id];
		devCudaOffsetBuffer[3 * id + 1] = positionY[id];
		devCudaOffsetBuffer[3 * id + 2] = positionZ[id];
	}

	graphics::GLController::GLController() : particleModel("Models/Earth/low_poly_earth.fbx")
	{
		// register OpenGL buffer in CUDA
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaOffsetResource, particleModel.getCudaOffsetBuffer(), cudaGraphicsRegisterFlagsNone));

		// Create a directional light
		directionalLight = DirLight
		{
			{
				vec3(0.4f, 0.4f, 0.4f), vec3(1, 1, 1), vec3(1, 1, 1)
			},
			vec3(0, 0, 1.0f)
		};

		// Set up deferred shading
		// Set up OpenGL frame buffers
		glGenFramebuffers(1, &gBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

		unsigned int gPosition, gNormal;
		// position color buffer
		glGenTextures(1, &gPosition);
		glBindTexture(GL_TEXTURE_2D, gPosition);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
		// normal color buffer
		glGenTextures(1, &gNormal);
		glBindTexture(GL_TEXTURE_2D, gNormal);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
		
		// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
		unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, attachments);

		// create and attach depth buffer (renderbuffer)
		unsigned int rboDepth;
		glGenRenderbuffers(1, &rboDepth);
		glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

		// finally check if framebuffer is complete
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Create the shaders
		solidColorShader = std::make_shared<Shader>(SolidColorShader());
		geometryPassShader = std::make_shared<Shader>(GeometryPassShader(gBuffer));
		phongLightingShader = std::make_shared<Shader>(PhongLightingShader(gPosition, gNormal));
	}

	void graphics::GLController::calculateOffsets(float* positionX, float* positionY, float* positionZ, unsigned int particleCount)
	{
		// get CUDA a pointer to openGL buffer
		float* devCudaOffsetBuffer = 0;
		size_t numBytes;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaOffsetResource, 0));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devCudaOffsetBuffer, &numBytes, cudaOffsetResource));
		
		// translate our CUDA positions into Vertex offsets
		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
		calculateOffsetsKernel << <blocks, threadsPerBlock >> > (devCudaOffsetBuffer, positionX, positionY, positionZ, particleCount);

		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaOffsetResource, 0));
	}

	void graphics::GLController::draw()
	{
		if constexpr (!useLighting)
		{
			solidColorShader->use();
			solidColorShader->setMatrix("model", model);
			solidColorShader->setMatrix("view", view);
			solidColorShader->setMatrix("projection", projection);

			particleModel.draw(solidColorShader);
			return;
		}

		// Geometry pass

		geometryPassShader->use();
		geometryPassShader->setMatrix("model", model);
		geometryPassShader->setMatrix("view", view);
		geometryPassShader->setMatrix("projection", projection);

		glEnable(GL_DEPTH_TEST);

		particleModel.draw(geometryPassShader);

		// Phong Lighting Pass

		glDisable(GL_DEPTH_TEST);

		phongLightingShader->use();
		phongLightingShader->setVector("viewPos", cameraPosition);
		phongLightingShader->setVector("Diffuse", particleDiffuse);
		phongLightingShader->setFloat("Specular", particleSpecular);
		phongLightingShader->setFloat("Shininess", 32);

		phongLightingShader->setLighting(directionalLight);

		// copy content of geometry's depth buffer to default framebuffer's depth buffer
		// ----------------------------------------------------------------------------------
		glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
		// blit to default framebuffer. Note that this may or may not work as the internal formats of both the FBO and default framebuffer have to match.
		// the internal formats are implementation defined. This works on all of my systems, but if it doesn't on yours you'll likely have to write to the 		
		// depth buffer in another shader stage (or somehow see to match the default framebuffer's internal format with the FBO's internal format).
		glBlitFramebuffer(0, 0, windowWidth, windowHeight, 0, 0, windowWidth, windowHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Render quad
		unsigned int quadVAO = 0;
		unsigned int quadVBO = 0;

		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
				1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
				1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glBindVertexArray(quadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
		return;
	}
}