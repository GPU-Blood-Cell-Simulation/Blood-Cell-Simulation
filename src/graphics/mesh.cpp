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
	auto obj = vertices;
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

MultiObjectMesh::MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures, std::vector<glm::vec3>& initialPositions, unsigned int objectCount)
	: Mesh(std::move(vertices), std::move(indices), std::move(textures))
{
	this->objectCount = objectCount;
	prepareMultipleObjects(initialPositions);
	setupMesh();
}

void MultiObjectMesh::prepareMultipleObjects(std::vector<glm::vec3>& initialPositions)
{
	unsigned int indicesCount = indices.size() * initialPositions.size();
	std::vector<unsigned int> newIndices(indicesCount);
	for (int i = 0; i < initialPositions.size(); ++i) {
		std::transform(indices.cbegin(), indices.cend(), newIndices.begin() + i * indices.size(),
			[&](unsigned int indice) {

				// if switch order of transforming indices (after vertices), remember to handle this vertices.size()
				return indice + i * vertices.size();

			});
	}
	indices = std::vector<unsigned int>(indicesCount);
	std::move(newIndices.begin(), newIndices.end(), indices.begin());

	unsigned int verticesCount = vertices.size() * initialPositions.size();
	std::vector<Vertex> newVertices(verticesCount);
	for (int i = 0; i < initialPositions.size(); ++i) {
		std::transform(vertices.cbegin(), vertices.cend(),
			(newVertices.begin() + i*vertices.size()),
			[&](Vertex v) {

				Vertex v2;
				v2.normal = v.normal;
				v2.texCoords = v.texCoords;
				v2.position = v.position + initialPositions[i];
				return v2;

			});
	}
	vertices = std::vector<Vertex>(verticesCount);
	std::move(newVertices.begin(), newVertices.end(), vertices.begin());

}

void MultiObjectMesh::DuplicateObjects(std::vector<glm::vec3>& initialPositions)
{
	for (int i = 0; i < initialPositions.size(); ++i) {
		std::transform(vertices.cbegin(), vertices.cend(),
			(!i ? vertices.begin() : vertices.end()),
			[&](Vertex v) {
				Vertex v2;
				v2.normal = v.normal;
				v2.texCoords = v.texCoords;
				v2.position = v.position + initialPositions[i];
				return v2;
			});
	}

	for (int i = 1; i < initialPositions.size(); ++i) {
		std::transform(indices.cbegin(), indices.cend(), indices.end(),
			[&](unsigned int indice) {
				return indice + i * objectCount;
			});
	}
}

//
//PredefinedMesh::PredefinedMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices)
//{
//	/*this->cellIndex = index;
//	using BloodCellDefinition = mp_at_c<BloodCellList, cellIndex>;
//	_verticesCount = BloodCellDefinition::particlesInCell;
//	_indicesCount = 6 * _verticesCount;
//
//	using verticeIndexList = mp_iota_c<_verticesCount>;
//	using VerticeList = typename BloodCellDefinition::Vertices;
//	using NormalList = typename BloodCellDefinition::Normals;
//	mp_for_each<verticeIndexList>([&](auto i)
//		{
//			Vertex v = new Vertex();
//			v.position = glm::vec3(
//				mp_at_c<VerticeList, i>::x,
//				mp_at_c<VerticeList, i>::y,
//				mp_at_c<VerticeList, i>::z
//			);
//			v.normal = glm::vec3(
//				mp_at_c<NormalList, i>::x,
//				mp_at_c<NormalList, i>::y,
//				mp_at_c<NormalList, i>::z
//			);
//			vertices.push_back(v);
//		});
//
//	using indiceIndexList = mp_iota_c<_indicesCount>;
//	using IndiceList = typename BloodCellDefinition::Indices;
//	mp_for_each<indiceIndexList>([&](auto i) {
//		indices.push_back(mp_at_c<IndiceList, i>);
//	});*/
//	this->vertices = std::move(vertices);
//}
//
//void PredefinedMesh<index>::DuplicateObjects(std::vector<glm::vec3>& initialPositions)
//{
//	for (int i = 0; i < initialPositions.size(); ++i) {
//		std::transform(vertices.begin(), vertices.end(),
//			(!i ? vertices.begin() : vertices.end()),
//			[&](auto& v) {
//				v->position += initialPositions[i];
//			});
//	}
//
//	for (int i = 1; i < initialPositions.size(); ++i) {
//		std::transform(indices.begin(), indices.end(), indices.end(),
//			[&](auto& indice) {
//				indice += i * _verticesCount;
//			});
//	}
//}
