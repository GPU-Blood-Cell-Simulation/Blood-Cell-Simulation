#pragma once

#include "model.hpp"
#include "camera.hpp"
#include "inputcontroller.hpp"
#include "light.hpp"
#include "../defines.cuh"

#include <memory>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace graphics
{
	// Controls rendering of the particles
	class GLController {
	public:

		explicit GLController(GLFWwindow* window);
		void calculateOffsets(float* positionX, float* positionY, float* positionZ, unsigned int particleCount);
		void draw();
		inline void handleInput()
		{
			inputController.adjustParametersUsingInput(camera);
		}

	private:

		// Particle color
		glm::vec3 particleDiffuse = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;

		// Uniform matrices
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));
		glm::mat4 projection = glm::perspective(glm::radians<float>(45.0f), windowWidth / windowHeight, 0.1f, depth * 10);

		Model particleModel = Model("Models/Earth/low_poly_earth.fbx");
    Model veinModel = Model("Models/Cylinder/cylinder.obj");

		Camera camera;
		InputController inputController;

		DirLight directionalLight;

		std::shared_ptr<Shader> solidColorShader;
		std::shared_ptr<Shader> geometryPassShader;
		std::shared_ptr<Shader> phongDeferredShader;
		std::shared_ptr<Shader> phongForwardShader;
		std::shared_ptr<Shader> cylinderSolidColorShader;
		
		unsigned int gBuffer;

		cudaGraphicsResource_t cudaOffsetResource;

	};
}