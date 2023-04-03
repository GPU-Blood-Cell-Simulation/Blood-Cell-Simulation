#pragma once

#include <string>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

// https://learnopengl.com/Getting-started/Shaders
class Shader
{
public:
    // the program ID
    unsigned int ID;
    // use/activate the shader
    virtual ~Shader() = default;
    virtual void use();
    // utility uniform functions
    void setFloat(const char* name, float value) const;
    void setInt(const char* name, int value) const;
    void setVector(const char* name, glm::vec3& vector) const;
    void setMatrix(const char* name, glm::mat4& matrix) const;
protected:
    // constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath);
};

class SolidColorShader : public Shader
{
public:
    SolidColorShader();
};