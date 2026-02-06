#pragma once
#include <memory>
#include <vector>
#include <stdexcept>

#include "rays.h" // must see the Rays class

static inline std::unique_ptr<Rays> MakeFilteredRays(const Rays *src, const std::vector<int> &idx)
{
    if (!src)
        error("MakeFilteredRays: src is null.");

    const auto &dirs = src->GetRayDirections(); // (K x 3)
    const auto &omegas = src->GetSolidAngles(); // (K)

    fArray2D outDirs;
    fArray1D outOmegas;
    outDirs.reserve(idx.size());
    outOmegas.reserve(idx.size());

    for (int j : idx)
    {
        outDirs.push_back(dirs.at(static_cast<size_t>(j)));
        outOmegas.push_back(omegas.at(static_cast<size_t>(j)));
    }

    return std::make_unique<Rays>(outDirs, outOmegas);
}