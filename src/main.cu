#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "defines.hpp"
#include "utilities/cuda_handle_error.cuh"

#include "simulation/simulation.cuh"
#include "graphics/glcontroller.cuh"
#include "grids/uniform_grid.cuh"
#include "grids/no_grid.cuh"

#include "objects/blood_cells.cuh"
#include "objects/blood_cells_factory.hpp"
#include "objects/device_triangles.cuh"
#include "objects/cylindermesh.hpp"

#include <GLFW/glfw3.h>
#include <sstream>

#include <curand.h>
#include <curand_kernel.h>

//#pragma float_control( except, on )
//// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
//extern "C"
//{
//    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
//}

void programLoop(GLFWwindow* window);

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
    VEIN_POLYGON_MODE = GL_FILL;

    // debug
    glEnable(GL_DEBUG_OUTPUT);
    
#pragma endregion
    
    // Main simulation loop
    
    programLoop(window);

    // Cleanup

    glfwTerminate();
    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}

// Main simulation loop - upon returning from this function all memory-freeing destructors are called
void programLoop(GLFWwindow* window)
{
    double lastTime = glfwGetTime();
    int frameCount = 0;

    // Create dipols
    BloodCells bloodCells = BloodCellsFactory::createDipols(PARTICLE_COUNT / 2, springsInCellsLength);

    // Create vein mesh
    CylinderMesh veinMeshDefinition(cylinderBaseCenter, cylinderHeight, cylinderRadius, cylinderVerticalLayers, cylinderHorizontalLayers);
    Mesh veinMesh = veinMeshDefinition.CreateMesh();

    // Create vein triangles
    DeviceTriangles triangles(veinMesh, veinMeshDefinition.getSpringLengths());

    // Create grids
    UniformGrid particleGrid(PARTICLE_COUNT, 20, 20, 20);
    UniformGrid triangleCentersGrid(triangles.triangleCount, 10, 10, 10);
    //NoGrid particleGrid, triangleCentersGrid;

    // Create the main simulation controller and inject its dependencies
    sim::SimulationController simulationController(bloodCells, triangles, &particleGrid, &triangleCentersGrid);

    // Create a graphics controller
    graphics::GLController glController(window, veinMesh);

    // MAIN LOOP HERE - dictated by glfw

    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate particle positions using CUDA
        simulationController.calculateNextFrame();


        // Pass positions to OpenGL
        glController.calculateOffsets(bloodCells.particles.position.x,
            bloodCells.particles.position.y,
            bloodCells.particles.position.z,
            bloodCells.particleCount);
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
}