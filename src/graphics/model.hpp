#pragma once

#include <assimp/scene.h>
#include <memory>
#include <string>
#include <vector>

#include "mesh.hpp"
#include "shader.hpp"


class Model
{
public:
    Model(const char* path);
    Model(Mesh* mesh);

    virtual void draw(const Shader* shader) = 0;

    Mesh* getMesh(unsigned int index);
    unsigned int getVboBuffer(unsigned int index);
    unsigned int getEboBuffer(unsigned int index);
protected:
    

    std::vector<Texture> textures_loaded;
    std::string directory;
    std::vector<Mesh*> meshes;

    void loadModel(std::string path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh* processMesh(aiMesh* mesh, const aiScene* scene);

    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
    virtual Mesh* createMesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures) = 0;

};

class SingleObjectModel : public Model
{
public:
    SingleObjectModel(const char* path) : Model(path) {}
    SingleObjectModel(Mesh* mesh) : Model(mesh) {}

    void draw(const Shader* shader) override;
protected:
    Mesh* createMesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures) override;
};

class InstancedModel : public Model
{
public:
    InstancedModel(const char* path, unsigned int instancesCount);
    InstancedModel(Mesh* mesh, unsigned int instancesCount);

    void draw(const Shader* shader) override;
    unsigned int getCudaOffsetBuffer();
    
protected:

    // Array of translation vectors for each instance - cuda writes to this
    unsigned int cudaOffsetBuffer;
    unsigned int instancesCount;

    Mesh* createMesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures) override;
};

class MultipleObjectModel : public Model
{
public:
    MultipleObjectModel(const char* path, unsigned int objectCount);
    //MultipleObjectModel(Mesh* mesh, unsigned int objectCount);

    void draw(const Shader* shader) override;

protected:
    unsigned int objectCount;
    std::vector<glm::vec3> initialPositions;

    Mesh* createMesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures) override;
};