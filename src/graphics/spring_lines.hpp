#pragma once

#include "../graphics/shader.hpp"

#include <memory>
#include <vector>

class SpringLines
{
public:
	SpringLines(std::vector<unsigned int>&& indexDataTemplate, unsigned int VBO);
	void draw(const Shader* shader) const;
private:
	std::vector<unsigned int> indexData;
	unsigned int VAO;
};