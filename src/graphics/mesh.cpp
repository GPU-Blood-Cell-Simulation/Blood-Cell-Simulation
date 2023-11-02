#include "mesh.hpp"

#include "../meta_factory/blood_cell_factory.hpp"

#include <glad/glad.h>
#include <memory>
#include <algorithm>

Mesh::Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures) :
	vertices(vertices), indices(indices), textures(textures) {}

void Mesh::setupMesh()
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int),
		&indices[0], GL_STATIC_DRAW);

	// vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	// vertex texture coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

	glBindVertexArray(0);
}

unsigned int Mesh::getVBO()
{
	return VBO;
}

unsigned int Mesh::getEBO()
{
	return EBO;
}

void Mesh::setVertexOffsetAttribute()
{
	glBindVertexArray(VAO);
	// instance offset vectors
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glVertexAttribDivisor(3, 1);

	glBindVertexArray(0);
}

SingleObjectMesh::SingleObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures)
	: Mesh(std::move(vertices), std::move(indices), std::move(textures))
{
	setupMesh();
}

void Mesh::draw(const Shader* shader) const
{
	// draw mesh
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void SingleObjectMesh::drawInstanced(const Shader* shader, unsigned int instancesCount) const
{
	// draw mesh
	glBindVertexArray(VAO);
	glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0, particleCount);
	glBindVertexArray(0);
}

MultiObjectMesh::MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures, unsigned int objectCount)
	: Mesh(std::move(vertices), std::move(indices), std::move(textures))
{
	this->objectCount = objectCount;
	prepareMultipleObjects();
	setupMesh();
}

void MultiObjectMesh::prepareMultipleObjects()
{
	for (int i = 0; i < objectCount; ++i) {
		std::copy(vertices.begin(), vertices.end(), vertices.end());
		std::copy(indices.begin(), indices.end(), indices.end());
	}
	for (int i = 1; i < objectCount; ++i) {
		std::for_each(indices.begin(), indices.end(), [](auto& indice) {
			indice += i * objectCount;
		});
	}
}
