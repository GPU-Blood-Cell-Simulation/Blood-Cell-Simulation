#pragma once

namespace sim
{
    void createUniformGrid(const float* positionX, const float* positionY, const float* positionZ,
        unsigned int* cellIds, unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);
}
