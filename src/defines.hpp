#pragma once

inline constexpr int windowWidth = 800;
inline constexpr int windowHeight = 800;

inline constexpr float width = 200.0f;
inline constexpr float height = 200.0f;
inline constexpr float depth = 200.0f;
//constexpr float3 dimension {100,100,100};


// cylinder model data:
// min_x = -1.00156, max_x = 0.998437, min_y = -0.130239, max_y = 5.14687
inline constexpr float cylinderRadius = 1.0f;
inline constexpr float cylinderHeight = 5.27716f; // scale = glm::vec3(width / 2, 2 * width, width / 2);
inline constexpr float cylinderScaleX = width / 2;
inline constexpr float cylinderScaleY = height / 2;
inline constexpr float cylinderScaleZ = depth / 2;

inline constexpr int cellWidth = 20;
inline constexpr int cellHeight = 20;
inline constexpr int cellDepth = 20;

// 96 = 3*2*4*4
// three layers of 2 particle dipoles, each layer is 4x4
inline constexpr unsigned int PARTICLE_COUNT = 500;

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.1f;

// initial particle velocity value
inline constexpr float v0 = 0.1f;

/// used in allocating new particles in array
//inline unsigned int new_cell_index = 0;

/// PHYSICS CONST

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