#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "simulation.cuh"
#include "defines.cuh"
#include "objects.cuh"
#include "graphics/glcontroller.cuh"

#include "BloodCell/BloodCells.cuh"

#include <GLFW/glfw3.h>
#include <sstream>

#include <curand.h>
#include <curand_kernel.h>


int main()
{
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
    window = glfwCreateWindow(windowWidth, windowHeight, "Blood Cell Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Load GL and set the viewport to match window size
    gladLoadGL();
    glViewport(0, 0, windowWidth, windowHeight);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // debug
    glEnable(GL_DEBUG_OUTPUT);

    double lastTime = glfwGetTime();
    int frameCount = 0;
#pragma endregion

    // Create a graphics controller
    graphics::GLController glController(window);

    float graph[]{ 0, 5, 5, 5, 0, 5, 5, 5, 0 };
    BloodCells cells(1, 3, graph);

    // Generate random positions
    sim::generateRandomPositions(cells.particles, cells.particlesCnt);

    // MAIN LOOP HERE - probably dictated by glfw

    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate particle positions using CUDA
        sim::calculateNextFrame(cells);

        // Pass positions to OpenGL
        glController.calculateOffsets(cells.particles.position.x,
                                      cells.particles.position.y,
                                      cells.particles.position.z,
                                      cells.particlesCnt);

        // OpenGL render
#pragma region rendering
        
        glController.draw();
        glfwSwapBuffers(window);

        // Show FPS in the title bar
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        if (delta >= 1.0)
        {
            double fps = double(frameCount) / delta;
            std::stringstream ss;
            ss << "Blood Cell Simulation" << " " << " [" << fps << " FPS]";

            glfwSetWindowTitle(window, ss.str().c_str());
            lastTime = currentTime;
            frameCount = 0;
        }
        else
        {
            frameCount++;
        }
#pragma endregion

        // Handle user input
        glfwPollEvents();
        glController.handleInput();
    }

    // Cleanup
    cells.Deallocate();

    glfwTerminate();

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}