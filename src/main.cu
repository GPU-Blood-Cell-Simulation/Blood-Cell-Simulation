#include "simulation.cuh"
#include "defines.cuh"
#include "graphics/glcontroller.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

int main()
{
    constexpr unsigned int particleCount = 500;

    float* positionX = 0;
    float* positionY = 0;
    float* positionZ = 0;
    unsigned int* cellIds = 0;
    unsigned int* particleIds = 0;
    unsigned int* cellStarts = 0;
    unsigned int* cellEnds = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // OpenGL setup
#pragma region OpenGLsetup
    GLFWwindow* window;

    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(windowWidth, windowHeight, "Boids", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Load GL and set the viewport to match window size
    gladLoadGL();;
    glViewport(0, 0, windowWidth, windowHeight);

    double lastTime = glfwGetTime();
    int frameCount = 0;
#pragma endregion

    // Create a graphics controller
    graphics::GLController glController;

    // Allocate memory
    sim::allocateMemory(&positionX, &positionY, &positionZ, &cellIds, &particleIds, &cellStarts, &cellEnds, particleCount);

    // Generate random positions
    sim::generateRandomPositions(positionX, positionY, positionZ, particleCount);

    // MAIN LOOP HERE - probably dictaded by glfw

    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Calculate particle positions using CUDA
        sim::calculateNextFrame(positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);

#pragma region rendering
        // OpenGL render
        glController.draw();
        glfwSwapBuffers(window);

        // Show FPS in the title bar
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        if (delta >= 1.0)
        {
            double fps = double(frameCount) / delta;
            std::stringstream ss;
            ss << "Boids" << " " << " [" << fps << " FPS]";

            glfwSetWindowTitle(window, ss.str().c_str());
            lastTime = currentTime;
            frameCount = 0;
            /*seconds++;
            if (seconds >= 5)
                exit(0);*/
        }
        else
        {
            frameCount++;
        }

        glfwPollEvents();
#pragma endregion
    }

    // Cleanup
    cudaFree(positionX);
    cudaFree(positionY);
    cudaFree(positionZ);
    cudaFree(cellIds);
    cudaFree(particleIds);
    cudaFree(cellStarts);
    cudaFree(cellEnds);

    glfwTerminate();

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}