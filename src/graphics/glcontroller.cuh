#pragma once

#include "model.hpp"
#include "camera.hpp"
#include "inputcontroller.hpp"
#include "light.hpp"
#include "../defines.hpp"
#include "../blood_cell_structures/device_triangles.cuh"

#include <memory>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include "cylindermesh.hpp"

namespace graphics
{
	// Controls rendering of the particles
	class GLController {
	public:

		explicit GLController(GLFWwindow* window);
		void calculateOffsets(float* positionX, float* positionY, float* positionZ, unsigned int particleCount);
		void calculateTriangles(DeviceTriangles triangles);
		void draw();
		inline void handleInput()
		{
			inputController.adjustParametersUsingInput(camera);
		}

		Mesh getGridMesh()
		{
			return veinModel.getTopMesh();
		}

	private:

		// Particle color
		glm::vec3 particleDiffuse = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;

		// Uniform matrices
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));
		glm::mat4 projection = glm::perspective(glm::radians<float>(45.0f), static_cast<float>(windowWidth) / windowHeight, 0.1f, depth * 10);

		Model particleModel = Model("Models/Earth/low_poly_earth.fbx");
		Model veinModel = Model(CylinderMesh(cylinderBaseCenter, cylinderHeight, cylinderRadius, 
			cylinderVerticalLayers, cylinderHorizontalLayers).CreateMesh());

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
		cudaGraphicsResource_t cudaVeinVBOResource;
		cudaGraphicsResource_t cudaVeinEBOResource;

	};
}