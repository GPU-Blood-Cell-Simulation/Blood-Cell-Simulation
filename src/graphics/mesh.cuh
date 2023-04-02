#pragma once

#include <vector>
#include <memory>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include "shader.hpp"

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
};


class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices);
    void draw(const std::shared_ptr<Shader> shader);
private:
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};