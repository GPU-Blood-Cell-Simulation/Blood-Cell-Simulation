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

		glm::vec3 particleDiffuse = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;
		glm::vec3 cameraPosition = glm::vec3(0, 0, 50);
		glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
		glm::mat4 projection = glm::perspective(60.0f, windowWidth/windowHeight, 0.1f,  200.0f);

		Model particleModel;
		DirLight directionalLight;
		std::shared_ptr<Shader> solidColorShader;
		std::shared_ptr<Shader> geometryPassShader;
		std::shared_ptr<Shader> phongLightingShader;
		cudaGraphicsResource_t VBOresource;

		unsigned int gBuffer;
	};
}