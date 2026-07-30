#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <cstdint>
#include <span>
#include <type_traits>

namespace blackframe::renderer {

// This is the deliberately narrow resolved surface accepted by the scalar BSDF-only oracle. It
// does not introduce a scene, material, light-sampling, or acceleration interface.
template <SpectrumScalar Scalar> class BsdfOnlyTriangleSurfaceT final {
  public:
    constexpr BsdfOnlyTriangleSurfaceT(TriangleT<Scalar> triangle,
                                       LambertianReflectionT<Scalar> reflection,
                                       OneSidedSurfaceEmissionT<Scalar> emission,
                                       SampledWavelengthsT<Scalar> wavelengths,
                                       const RayMask visibility_mask) noexcept
        : triangle_{triangle}, reflection_{reflection}, emission_{emission},
          wavelengths_{wavelengths}, visibility_mask_{visibility_mask} {}

    [[nodiscard]] constexpr const TriangleT<Scalar>& triangle() const noexcept {
        return triangle_;
    }

    [[nodiscard]] constexpr const LambertianReflectionT<Scalar>& reflection() const noexcept {
        return reflection_;
    }

    [[nodiscard]] constexpr const OneSidedSurfaceEmissionT<Scalar>& emission() const noexcept {
        return emission_;
    }

    [[nodiscard]] constexpr const SampledWavelengthsT<Scalar>& wavelengths() const noexcept {
        return wavelengths_;
    }

    [[nodiscard]] constexpr RayMask visibility_mask() const noexcept {
        return visibility_mask_;
    }

  private:
    TriangleT<Scalar> triangle_;
    LambertianReflectionT<Scalar> reflection_;
    OneSidedSurfaceEmissionT<Scalar> emission_;
    SampledWavelengthsT<Scalar> wavelengths_;
    RayMask visibility_mask_;
};

using BsdfOnlyTriangleSurface = BsdfOnlyTriangleSurfaceT<TransportScalar>;
using ReferenceBsdfOnlyTriangleSurface = BsdfOnlyTriangleSurfaceT<ReferenceScalar>;

// Environment radiance is resolved at exactly the same wavelength packet as every surface. The
// loop rejects a packet mismatch instead of reinterpreting fixed lanes at unrelated wavelengths.
template <SpectrumScalar Scalar> class BsdfOnlyEnvironmentT final {
  public:
    constexpr BsdfOnlyEnvironmentT(ConstantEnvironmentT<Scalar> environment,
                                   SampledWavelengthsT<Scalar> wavelengths) noexcept
        : environment_{environment}, wavelengths_{wavelengths} {}

    [[nodiscard]] constexpr const ConstantEnvironmentT<Scalar>& environment() const noexcept {
        return environment_;
    }

    [[nodiscard]] constexpr const SampledWavelengthsT<Scalar>& wavelengths() const noexcept {
        return wavelengths_;
    }

  private:
    ConstantEnvironmentT<Scalar> environment_;
    SampledWavelengthsT<Scalar> wavelengths_;
};

using BsdfOnlyEnvironment = BsdfOnlyEnvironmentT<TransportScalar>;
using ReferenceBsdfOnlyEnvironment = BsdfOnlyEnvironmentT<ReferenceScalar>;

enum class BsdfOnlyPathTermination : std::uint8_t {
    escaped_environment,
    maximum_depth,
    outside_bsdf_support,
    zero_throughput,
};

template <SpectrumScalar Scalar> struct BsdfOnlyPathResultT final {
    PathStateT<Scalar> state;
    RayT<Scalar> terminal_ray;
    BsdfOnlyPathTermination termination;
};

using BsdfOnlyPathResult = BsdfOnlyPathResultT<TransportScalar>;
using ReferenceBsdfOnlyPathResult = BsdfOnlyPathResultT<ReferenceScalar>;

// Traces Lambertian continuation rays through a linearly searched triangle span. Emission is
// accumulated at every hit and the required constant environment is accumulated on a miss. There
// is intentionally no next-event estimation, light sampling, MIS, Russian roulette, or fallback
// backend. maximum_depth counts accepted scattering events; surface emission remains visible at
// the cutoff. Secondary rays use [0, +infinity] while preserving time, mask, and medium.
[[nodiscard]] core::Result<BsdfOnlyPathResult>
trace_bsdf_only(const Ray& initial_ray, const PathState& initial_state,
                const SampleStream& sample_stream,
                std::span<const BsdfOnlyTriangleSurface> surfaces,
                const BsdfOnlyEnvironment& environment, std::uint32_t maximum_depth);

[[nodiscard]] core::Result<ReferenceBsdfOnlyPathResult>
trace_bsdf_only(const ReferenceRay& initial_ray, const ReferencePathState& initial_state,
                const ReferenceSampleStream& sample_stream,
                std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces,
                const ReferenceBsdfOnlyEnvironment& environment, std::uint32_t maximum_depth);

static_assert(sizeof(BsdfOnlyPathTermination) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<BsdfOnlyTriangleSurface>);
static_assert(std::is_trivially_copyable_v<BsdfOnlyTriangleSurface>);
static_assert(std::is_standard_layout_v<ReferenceBsdfOnlyTriangleSurface>);
static_assert(std::is_trivially_copyable_v<ReferenceBsdfOnlyTriangleSurface>);
static_assert(std::is_standard_layout_v<BsdfOnlyEnvironment>);
static_assert(std::is_trivially_copyable_v<BsdfOnlyEnvironment>);
static_assert(std::is_standard_layout_v<ReferenceBsdfOnlyEnvironment>);
static_assert(std::is_trivially_copyable_v<ReferenceBsdfOnlyEnvironment>);
static_assert(std::is_standard_layout_v<BsdfOnlyPathResult>);
static_assert(std::is_trivially_copyable_v<BsdfOnlyPathResult>);
static_assert(std::is_standard_layout_v<ReferenceBsdfOnlyPathResult>);
static_assert(std::is_trivially_copyable_v<ReferenceBsdfOnlyPathResult>);

} // namespace blackframe::renderer
