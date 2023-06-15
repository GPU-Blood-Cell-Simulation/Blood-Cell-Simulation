#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "simulation.cuh"
#include "defines.cuh"
#include "objects.cuh"
#include "graphics/glcontroller.cuh"
#include "uniform_grid.cuh"

#include <GLFW/glfw3.h>
#include <sstream>

#include <curand.h>
#include <curand_kernel.h>
//// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
//extern "C"
//{
//    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
//}

int main()
{
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

    // Allocate memory
    Particles particles(PARTICLE_COUNT);
    Corpuscles corpscles = Corpuscles(10);
    UniformGrid grid, triangleCentersGrid;
    DeviceTriangles triangles = DeviceTriangles(glController.getGridMesh());

    sim::allocateMemory(grid, PARTICLE_COUNT);
    sim::allocateMemory(triangleCentersGrid, triangles.trianglesCount);
    triangleCentersGrid.calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangles.trianglesCount);

    // Generate random positions
    sim::generateRandomPositions(particles, PARTICLE_COUNT);
    //sim::generateInitialPositionsInLayers(particles, corpscles, PARTICLE_COUNT, 3);

    // MAIN LOOP HERE - probably dictated by glfw

    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate particle positions using CUDA
        sim::calculateNextFrame(particles, corpscles, triangles, grid, PARTICLE_COUNT, triangles.trianglesCount);

        // Pass positions to OpenGL
        glController.calculateOffsets(particles.position.x, particles.position.y, particles.position.z, PARTICLE_COUNT);
        glController.calculateTriangles(triangles);
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
    particles.freeParticles();

    glfwTerminate();

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}