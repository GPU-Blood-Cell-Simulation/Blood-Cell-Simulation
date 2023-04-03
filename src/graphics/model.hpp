#pragma once

#include <vector>
#include <memory>
#include <string>
#include <assimp/scene.h>

#include "shader.hpp"
#include "mesh.hpp"
class Model
{
public:
    Model(const char* path);

    void draw(const std::shared_ptr<Shader> shader) const;
    unsigned int getVBO();

protected:

    std::string directory;
    std::vector<Mesh> meshes;

    void loadModel(std::string path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);

};