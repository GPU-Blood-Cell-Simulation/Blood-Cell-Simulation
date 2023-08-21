#pragma once
#include<glm/vec3.hpp>

inline constexpr int windowWidth = 800;
inline constexpr int windowHeight = 800;

inline constexpr float width = 300.0f;
inline constexpr float height = 500.0f;
inline constexpr float depth = 300.0f;

// cylinder model data:
inline constexpr float cylinderHeight = 0.8 * height;
inline constexpr float cylinderRadius = 0.2 * width;
inline glm::vec3 cylinderBaseCenter = glm::vec3(width / 2.0f, 0.1f * height, depth / 2.0f);
inline constexpr unsigned int cylinderVerticalLayers = 100;
inline constexpr unsigned int cylinderHorizontalLayers = 30;

// blood cell parameters
inline constexpr unsigned int PARTICLE_COUNT = 500;
inline constexpr float springsInCellsLength = 10;

// debug
inline unsigned int FRAME = 0;
inline unsigned int VEIN_POLYGON_MODE = 0;

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.1f;

// initial particle velocity value
inline constexpr float v0 = 0.1f;

// distance beetwen particle and vein on which an impact occurs
inline constexpr float veinImpactDistance = 5.0f;

/// used in allocating new particles in array
//inline unsigned int new_cell_index = 0;

/// PHYSICS CONST

// factor to slow down particles after collision
inline constexpr float velocityCollisionDamping = 0.8f;

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
inline constexpr float k_sniff = 0.1f;

// ! this value should be determined experimentally !
// Damping factor 
inline constexpr float d_fact = 0.1f;

// Lighting
inline constexpr bool useLighting = true;

// Camera movement constants
inline constexpr float cameraMovementSpeed = width / 100;
inline constexpr float cameraRotationSpeed = 0.02f;
