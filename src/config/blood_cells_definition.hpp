#pragma once

// blood cell parameters
inline constexpr unsigned int particlesInBloodCell = 4;
inline constexpr unsigned int bloodCellsCount = 250;
inline constexpr unsigned int particleCount = bloodCellsCount * particlesInBloodCell;

inline constexpr float springsInCellsLength = 10;
inline constexpr float particleRadius = 5;