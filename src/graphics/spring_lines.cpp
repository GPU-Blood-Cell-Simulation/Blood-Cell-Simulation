#include "spring_lines.hpp"

#include "../defines.hpp"

#include <glad/glad.h>

SpringLines::SpringLines(std::vector<unsigned int>&& indexDataTemplate, unsigned int VBO) : indexData(indexDataTemplate)
{
	if constexpr (particlesInBloodCell == 1)
		return;

	// multiply the index data
	int indexDataSize = indexData.size();
	for (int i = 1; i < bloodCellsCount; i++)
	{
		for (int j = 0; j < indexDataSize; j++)
		{
			indexData.push_back(i * particlesInBloodCell + indexData[j]);
		}
	}

	// setup VAO and EBO (VBO is shared with cuda-mapped position buffer
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
			unsigned int EBO;
			glGenBuffers(1, &EBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData.size() * sizeof(unsigned int), indexData.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

void SpringLines::draw(const Shader* shader) const
{
	if constexpr (particlesInBloodCell == 1)
		return;

	glBindVertexArray(VAO);
		glDrawElements(GL_LINES, static_cast<GLsizei>(indexData.size()), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}
