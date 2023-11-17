#include "model.hpp"

#include "../meta_factory/blood_cell_factory.hpp"
#include "textures/texture_loading.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <glad/glad.h>
#include <iostream>
#include <vector>


#pragma region Model

Model::Model(const char* path)
{
    loadModel(path);
    
    // Set up vertex attrubute
    for (Mesh* mesh : meshes)
    {
        mesh->setVertexOffsetAttribute();
    }
}

Model::Model(Mesh* mesh)
{
    meshes.push_back(mesh);

    // Set up vertex attrubute
    for (Mesh* mesh : meshes)
    {
        mesh->setVertexOffsetAttribute();
    }
}

unsigned int Model::getVboBuffer(unsigned int index)
{
    if(index < meshes.size())
        return meshes[index]->getVBO();
    return 0;
}

unsigned int Model::getEboBuffer(unsigned int index)
{
    if (index < meshes.size())
        return meshes[index]->getEBO();
    return 0;
}


Mesh* Model::getMesh(unsigned int index)
{
    if (index < meshes.size())
        return meshes[index];
    return nullptr;
}


void Model::loadModel(std::string path)
{
    Assimp::Importer import;
    const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));

    // Process assimp nodes to load all Meshes
    processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode* node, const aiScene* scene)
{
    // process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    // then do the same for each of its children
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(node->mChildren[i], scene);
    }
}

Mesh* Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
    // data to fill
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    // walk through each of the mesh's vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex vertex;
        glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
        // positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.position = vector;
        // normals
        if (mesh->HasNormals())
        {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;
            vertex.normal = vector;
        }
        // texture coordinates
        if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {
            glm::vec2 vec;
            // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
            // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.texCoords = vec;
        }
        else
            vertex.texCoords = glm::vec2(0.0f, 0.0f);

        vertices.push_back(vertex);
    }

    // now walk through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        // retrieve all indices of the face and store them in the indices vector
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    // we assume a convention for sampler names in the shaders. Each diffuse texture should be named
    // as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
    // Same applies to other texture as the following list summarizes:
    // diffuse: texture_diffuseN
    // specular: texture_specularN
    // normal: texture_normalN

    // 1. diffuse maps
    std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "diffuse");
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
    // 2. specular maps
    std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "specular");
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

    // return a mesh object created from the extracted mesh data
    return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
    //return this->createMesh(std::move(vertices), std::move(indices), std::move(textures));
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName)
{
    std::vector<Texture> textures;
    for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
    {
        aiString str;
        mat->GetTexture(type, i, &str);
        bool skip = false;
        for (unsigned int j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
            {
                textures.push_back(textures_loaded[j]);
                skip = true;
                break;
            }
        }
        if (!skip)
        {   // if texture hasn't been loaded already, load it
            Texture texture;
            texture.id = stbi::textureFromFile(str.C_Str(), directory);
            texture.type = typeName;
            texture.path = str.C_Str();
            textures.push_back(texture);
            textures_loaded.push_back(texture); // add to loaded textures
        }
    }
    return textures;
}
#pragma endregion

#pragma region Instanced model

void InstancedModel::draw(const Shader* shader)
{
    for (Mesh* mesh : meshes)
        ((SingleObjectMesh*)mesh)->drawInstanced(shader, this->instancesCount);
}

unsigned int InstancedModel::getCudaOffsetBuffer()
{
    return cudaOffsetBuffer;
}

//Mesh* InstancedModel::createMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures)
//{
//    return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
//}

InstancedModel::InstancedModel(const char* path, unsigned int instancesCount): Model(path)
{
    createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
        return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
    };

    loadModel(path);

    this->instancesCount = instancesCount;
    glGenBuffers(1, &cudaOffsetBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
    glBufferData(GL_ARRAY_BUFFER, instancesCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    //glm::vec3 origin = meshes[0]->vertices[0].position;
    // Set up vertex attrubute
    for (Mesh* mesh : meshes)
    {
        mesh->setVertexOffsetAttribute();
        //mesh.vertices
    }
}

InstancedModel::InstancedModel(Mesh* mesh, unsigned int instancesCount) : Model(mesh)
{
    createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
        return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
    };

    meshes.push_back(mesh);
    this->instancesCount = instancesCount;
    glGenBuffers(1, &cudaOffsetBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
    glBufferData(GL_ARRAY_BUFFER, instancesCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    // Set up vertex attrubute
    for (Mesh* mesh : meshes)
    {
        mesh->setVertexOffsetAttribute();
    }
}
#pragma endregion

#pragma region Multiple object model

//MultipleObjectModel::MultipleObjectModel(const char* path, unsigned int objectCount): Model(path)
//{
//    createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
//        return new MultiObjectMesh(std::move(vertices), std::move(indices), std::move(textures), objectCount);
//        };
//
//    loadModel(path);
//
//    this->objectCount = objectCount;
//    glGenBuffers(1, &cudaOffsetBuffer);
//    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
//    glBufferData(GL_ARRAY_BUFFER, objectCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
//
//    // Set up vertex attrubute
//    for (Mesh* mesh : meshes)
//    {
//        mesh->setVertexOffsetAttribute();
//    }
//}

//MultipleObjectModel::MultipleObjectModel(Mesh* mesh, unsigned int objectCount): Model(mesh)
//{
//    createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
//        return new MultiObjectMesh(std::move(vertices), std::move(indices), std::move(textures), objectCount);
//        };
//    this->objectCount = objectCount;
//}


MultipleObjectModel::MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount)
{
    createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
            return mesh;
        };
    this->mesh = new MultiObjectMesh(std::move(vertices), std::move(indices), std::move(std::vector<Texture>()), initialPositions, objectCount);
    this->objectCount = objectCount;
    this->modelVerticesCount = mesh->vertices.size();

    meshes.push_back(mesh);
    glGenBuffers(1, &cudaOffsetBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
    glBufferData(GL_ARRAY_BUFFER, objectCount * modelVerticesCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    /*for (Mesh* mesh : meshes)
    {
        mesh->setVertexOffsetAttribute();
    }*/
}

void MultipleObjectModel::DuplicateObjects(std::vector<glm::vec3>& initialPositions)
{

    this->mesh->DuplicateObjects(initialPositions);
}

void MultipleObjectModel::draw(const Shader* shader)
{
    for (Mesh* mesh : meshes)
        mesh->draw(shader);
}

unsigned int MultipleObjectModel::getCudaOffsetBuffer()
{
    return cudaOffsetBuffer;
}

#pragma endregion

#pragma region Single object Model

void SingleObjectModel::draw(const Shader* shader)
{
    for (Mesh* mesh : meshes)
        mesh->draw(shader);
}

#pragma endregion