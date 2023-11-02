#pragma once

#include "../graphics/mesh.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>

class CylinderMesh
{
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	const float radius, height;
	const unsigned int vLayers, hLayers;
	glm::vec3 basisOrigin;

	std::tuple<float, float, float> springLengths;

public:
	CylinderMesh(glm::vec3 basisOrigin, float height, float radius,
		unsigned int cylinderVerticalLayers, unsigned int cylinderHorizontalLayers) : basisOrigin(basisOrigin),
		radius(radius), height(height), vLayers(cylinderVerticalLayers), hLayers(cylinderHorizontalLayers)
	{
				float triangleH = height / vLayers;
		float radianBatch = 2 * glm::pi<float>() / hLayers;
		float triangleBase = radianBatch * radius;
		for (unsigned int i = 0; i < vLayers; ++i)
		{
			float h = i * triangleH /*- height / 2*/;
			for (unsigned int j = 0; j < hLayers; ++j)
			{
				glm::vec3 position = glm::vec3(radius * cos(j * radianBatch),
					h, radius * sin(j * radianBatch)) + basisOrigin;
				Vertex v
				{
					position,
					glm::vec3(glm::normalize(position - glm::vec3(basisOrigin.x, basisOrigin.y, basisOrigin.z + h))),
					glm::vec2(0) // no textures
				};

				vertices.push_back(v);
				if (i < vLayers - 1)
				{
					int nextj = (j + 1) % hLayers;
					indices.push_back((i + 1) * hLayers + j);
					indices.push_back(i * hLayers + j);
					indices.push_back(i * hLayers + nextj);
					indices.push_back((i + 1) * hLayers + j);
					indices.push_back(i * hLayers + nextj);
					indices.push_back((i + 1) * hLayers + nextj);
				}
			}
		}
		// Vertex distance debug info
		/*
		std::cout << "Horizontal (j): " << length(vertices[0].Position - vertices[1].Position) << std::endl;
		std::cout << "Vertical (i): " << length(vertices[0].Position - vertices[hLayers].Position) << std::endl;
		std::cout << "Diagonal: " << length(vertices[0].Position - vertices[hLayers + 1].Position) << std::endl;
		exit(0);*/

		springLengths = std::make_tuple(
			length(vertices[0].position - vertices[1].position),
			length(vertices[0].position - vertices[hLayers].position),
			length(vertices[0].position - vertices[hLayers + 1].position)
		);
	}

	std::tuple<float, float, float> getSpringLengths()
	{
		return springLengths;
	}

	SingleObjectMesh* CreateMesh()
	{
		return new SingleObjectMesh(std::move(vertices), std::move(indices),
			std::move(std::vector<Texture>(0)));
	}
	
};