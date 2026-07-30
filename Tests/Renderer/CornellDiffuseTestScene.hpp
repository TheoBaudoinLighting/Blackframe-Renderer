#pragma once

#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <expected>
#include <vector>

namespace blackframe::renderer::cornell_test {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using SurfaceFor = BsdfOnlyTriangleSurfaceT<Scalar>;

template <SpectrumScalar Scalar> using WavelengthsFor = SampledWavelengthsT<Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar>
spectrum(const std::array<Scalar, TransportSpectrumSampleCount> values) {
    return SpectrumFor<Scalar>{.values = values};
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(value);
    return result;
}

template <SpectrumScalar Scalar> [[nodiscard]] WavelengthsFor<Scalar> fixed_wavelengths() {
    const auto stream = IndependentSamplerT<Scalar>{0}.make_stream(0, 0, 0);
    return sample_visible_wavelengths(stream);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SurfaceFor<Scalar>>
make_surface(const Point3T<Scalar> vertex0, const Point3T<Scalar> vertex1,
             const Point3T<Scalar> vertex2, const SpectrumFor<Scalar>& reflectance,
             const SpectrumFor<Scalar>& radiance, const WavelengthsFor<Scalar>& wavelengths) {
    const auto triangle = TriangleT<Scalar>::create(vertex0, vertex1, vertex2);
    if (!triangle.has_value()) {
        return std::unexpected(triangle.error());
    }
    const auto reflection = LambertianReflectionT<Scalar>::create(reflectance);
    if (!reflection.has_value()) {
        return std::unexpected(reflection.error());
    }
    const auto emission = OneSidedSurfaceEmissionT<Scalar>::create(radiance);
    if (!emission.has_value()) {
        return std::unexpected(emission.error());
    }
    return SurfaceFor<Scalar>{*triangle, *reflection, *emission, wavelengths, AllRayVisibility};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
append_quad(std::vector<SurfaceFor<Scalar>>& surfaces, const Point3T<Scalar> vertex0,
            const Point3T<Scalar> vertex1, const Point3T<Scalar> vertex2,
            const Point3T<Scalar> vertex3, const SpectrumFor<Scalar>& reflectance,
            const SpectrumFor<Scalar>& radiance, const WavelengthsFor<Scalar>& wavelengths) {
    const auto first = make_surface(vertex0, vertex1, vertex2, reflectance, radiance, wavelengths);
    if (!first.has_value()) {
        return std::unexpected(first.error());
    }
    const auto second = make_surface(vertex0, vertex2, vertex3, reflectance, radiance, wavelengths);
    if (!second.has_value()) {
        return std::unexpected(second.error());
    }
    surfaces.push_back(*first);
    surfaces.push_back(*second);
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::vector<SurfaceFor<Scalar>>>
make_cornell_surfaces(const WavelengthsFor<Scalar>& wavelengths) {
    const auto white = constant_spectrum(Scalar{0.75});
    const auto left_tint =
        spectrum<Scalar>({Scalar{0.75}, Scalar{0.25}, Scalar{0.25}, Scalar{0.2}});
    const auto right_tint =
        spectrum<Scalar>({Scalar{0.2}, Scalar{0.75}, Scalar{0.25}, Scalar{0.2}});
    const auto black = SpectrumFor<Scalar>{};
    const auto ceiling_radiance = spectrum<Scalar>({Scalar{8}, Scalar{6}, Scalar{4}, Scalar{2}});

    auto surfaces = std::vector<SurfaceFor<Scalar>>{};
    surfaces.reserve(10);
    for (const auto& status : std::array{
             // Back wall, +Z.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}}, white, black,
                         wavelengths),
             // Floor, +Y.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}}, white, black,
                         wavelengths),
             // Entire ceiling emitter, -Y. Its large area keeps this BSDF-only regression bounded.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}, .z = Scalar{2}}, black,
                         ceiling_radiance, wavelengths),
             // Left wall, +X.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{2}},
                         left_tint, black, wavelengths),
             // Right wall, -X.
             append_quad(surfaces, Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{2}},
                         Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}}, right_tint, black,
                         wavelengths),
         }) {
        if (!status.has_value()) {
            return std::unexpected(status.error());
        }
    }
    return surfaces;
}

} // namespace blackframe::renderer::cornell_test
