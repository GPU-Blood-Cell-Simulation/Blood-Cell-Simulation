#pragma once

#include "model.hpp"
#include "light.hpp"
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

		// particle color
		glm::vec3 particleDiffuse = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;

		glm::vec3 cameraPosition = glm::vec3(width/2, height/2, depth * 3);

		// uniform matrices
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));
		glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(width / 2, height / 2, depth / 2), glm::vec3(0, 1, 0));
		glm::mat4 projection = glm::perspective(glm::radians<float>(45.0f), windowWidth / windowHeight, 0.1f, depth * 10);

		Model particleModel;
		DirLight directionalLight;
		std::shared_ptr<Shader> solidColorShader;
		std::shared_ptr<Shader> geometryPassShader;
		std::shared_ptr<Shader> phongLightingShader;
		
		unsigned int gBuffer;

		cudaGraphicsResource_t cudaOffsetResource;
	};
}