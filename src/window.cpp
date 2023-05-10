#include "window.h"

#include <stdexcept>
#include <sstream>


GLFWwindow* Window::GetWindow(int width, int height)
{
    initiateGlfw();

    // Create a windowed mode window and its OpenGL context
    GLFWwindow*  window = glfwCreateWindow(width, height, "Blood Cell Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Cannot create a new window");
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

    return window;
}


Window::Window(int width, int height):
    window{ GetWindow(width, height) }, glController{}, lastTime{glfwGetTime()}
{
}


void Window::initiateGlfw()
{
    if (!glfwInit()) {
        throw std::runtime_error("GLFW init failed");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}


void Window::clear()
{
    glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


void Window::draw()
{
    glController.draw();
    glfwSwapBuffers(window);
}


void Window::calculateFPS()
{
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
}


Window::~Window()
{
    glfwTerminate();
}
