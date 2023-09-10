#pragma once

constexpr int ceilToInt(float num)
{
    return (static_cast<float>(static_cast<int>(num)) == num)
        ? static_cast<int>(num)
        : static_cast<int>(num) + ((num > 0) ? 1 : 0);
}