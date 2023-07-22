#pragma once

/// <summary>
/// Base grid class - uses the Curiously Recurring Template Pattern to provide compile-time inheritance. Use it with std::variant and std::visit
/// 
/// </summary>
/// <typeparam name="Derived">A concrete grid implementation</typeparam>
template<typename Derived>
class BaseGrid
{
protected:
    BaseGrid() {}
public:

    inline void calculateGrid(const Particles& particles)
    {
        static_cast<Derived*>(this)->calculateGrid(particles);
    }
    inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount)
    {
        static_cast<Derived*>(this)->calculateGrid(positionX, positionY, positionZ, particleCount);;
    }
};