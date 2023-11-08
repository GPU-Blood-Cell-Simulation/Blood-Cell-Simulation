#pragma once

#include "camera.hpp"
#include "../config/graphics.hpp"
#include "../config/simulation.hpp"
#include "inputcontroller.hpp"
#include "light.hpp"
#include "model.hpp"
#include "../objects/vein_triangles.cuh"
#include "../objects/cylindermesh.hpp"
#include "spring_lines.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>


namespace graphics
{
	// Controls rendering of the particles
	class GLController {
	public:

		GLController(GLFWwindow* window, Mesh* veinMesh, std::vector<glm::vec3>& initialPositions);
		void calculateOffsets(cudaVec3 positions);
		void calculatePositions(cudaVec3 positions);
		void calculateTriangles(VeinTriangles triangles);
		void draw();
		inline void handleInput()
		{
			inputController.adjustParametersUsingInput(camera);
		}

		Mesh* getGridMesh()
		{
			return veinModel.getMesh(0);
		}

	private:

		// Particle color
		glm::vec3 particleDiffuse = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;

		// Uniform matrices
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
		glm::mat4 projection = glm::perspective(glm::radians<float>(45.0f), static_cast<float>(windowWidth) / windowHeight, 0.1f, depth * 10);

		//Model particleModel = Model("Models/Earth/low_poly_earth.fbx");
		
		// instanced vertices commented
		// InstancedModel particleModel = InstancedModel("Models/BloodCell/bloodcell.obj", particleCount);
		
		// old way
		//MultipleObjectModel bloodCellModel = MultipleObjectModel("Models/BloodCell/bloodcell.obj", bloodCellCount);

		// new way
		MultipleObjectModel bloodCellmodel;// = MultipleObjectModel(new PredefinedMesh<0>(), bloodCellCount);

		SingleObjectModel veinModel;

		Camera camera;
		InputController inputController;

		DirLight directionalLight;

		const SpringLines springLines;

		std::unique_ptr<Shader> solidColorShader;
		std::unique_ptr<Shader> geometryPassShader;
		std::unique_ptr<Shader> phongDeferredShader;
		std::unique_ptr<Shader> phongForwardShader;
		std::unique_ptr<Shader> cylinderSolidColorShader;
		std::unique_ptr<Shader> springShader;
		
		unsigned int gBuffer;

		//cudaGraphicsResource_t cudaOffsetResource;
		cudaGraphicsResource_t cudaPositionsResource;
		cudaGraphicsResource_t cudaVeinVBOResource;
		cudaGraphicsResource_t cudaVeinEBOResource;

	};
}