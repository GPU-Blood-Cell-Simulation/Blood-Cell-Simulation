#pragma once

#include "graphics/glcontroller.cuh"
#include "objects.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>


class Window
{
public:
	Window(int width, int height);
	~Window();

	inline int shouldClose() {
		return glfwWindowShouldClose(window);
	}

	void clear();

	inline void updateParticles(const particles& p) {
		glController.calculateOffsets(p.position.x, p.position.y, p.position.z, p.position.size);
 	}

	void draw();

	inline void handleEvents() {
		glfwPollEvents();
	}

	void calculateFPS();

private:
	GLFWwindow* window;
	graphics::GLController glController;

	double lastTime;
	int frameCount = 0;

	static void initiateGlfw();
	static GLFWwindow* GetWindow(int width, int height);
};
