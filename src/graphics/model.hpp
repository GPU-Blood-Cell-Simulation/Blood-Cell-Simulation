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
    Model(Mesh mesh);

    void draw(const std::shared_ptr<Shader> shader, bool instanced = true) const;
    unsigned int getCudaOffsetBuffer();
    Mesh getTopMesh();
protected:
    // Array of translation vectors for each instance - cuda writes to this
    unsigned int cudaOffsetBuffer;

    std::vector<Texture> textures_loaded;
    std::string directory;
    std::vector<Mesh> meshes;

    void loadModel(std::string path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);

    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);

};