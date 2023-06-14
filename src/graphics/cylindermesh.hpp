#include "mesh.hpp"
#include <fstream>
#include <iomanip>

class CylinderMesh
{
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	float radius, height;
	unsigned int vLayers, hLayers;
	glm::vec3 origin;
	const float PI = 3.14159f;
public:
	CylinderMesh(glm::vec3 origin, float height, float radius,
		unsigned int verticalLayers, unsigned int horizontalLayers)
	{
		this->origin = origin;
		this->radius = radius;
		this->height = height;
		vLayers = verticalLayers;		// |
		hLayers = horizontalLayers;		// _

		float triangleH = height / vLayers;
		float radianBatch = 2 * PI / hLayers;
		float triangleBase = radianBatch * radius;
		for (int i = 0; i < vLayers; ++i)
		{
			float h = i * triangleH - height / 2;
			for (int j = 0; j < hLayers; ++j)
			{
				glm::vec3 position = glm::vec3(radius * cos(j * radianBatch),
					h , radius * sin(j * radianBatch)) + origin;
				Vertex v; 
				v.Position = position;
				v.Normal = glm::vec3(glm::normalize(position - glm::vec3(origin.x, origin.y, origin.z + h)));
				v.TexCoords = glm::vec2(0); // no textures

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
	}

	Mesh CreateMesh()
	{
		return Mesh(std::move(vertices), std::move(indices), 
			std::move(std::vector<Texture>(0)));
	}
};