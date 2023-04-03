#pragma once

#include "model.cuh"
#include "../defines.cuh"

#include <memory>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace graphics
{
	class GLController {
	public:

		GLController();
		void calculateOffsets(float* positionX, float* positionY, float* positionZ, unsigned int particleCount);
		void draw();

	private:
		glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 100), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
		glm::mat4 projection = glm::perspective(60.0f, windowWidth/windowHeight, 0.1f,  200.0f);
		Model particleModel;
		std::shared_ptr<Shader> solidColorShader;
		cudaGraphicsResource_t VBOresource;
	};
}