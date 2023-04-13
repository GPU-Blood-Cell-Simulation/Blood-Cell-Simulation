#pragma once

#include <vector>
#include <memory>
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
    // offset - filled by CUDA
    glm::vec3 Offset = glm::vec3(0, 0, 0);
};

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};


class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures);
    void draw(const std::shared_ptr<Shader> shader);
    unsigned int getVBO();
private:
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};