#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
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
    depth_limit,
    outside_bsdf_support,
    zero_throughput,
    russian_roulette,
};

template <SpectrumScalar Scalar> struct BsdfOnlyPathResultT final {
    PathStateT<Scalar> state;
    RayT<Scalar> terminal_ray;
    BsdfOnlyPathTermination termination;
    ScatteringLobe blocked_depth_limits;
};

using BsdfOnlyPathResult = BsdfOnlyPathResultT<TransportScalar>;
using ReferenceBsdfOnlyPathResult = BsdfOnlyPathResultT<ReferenceScalar>;

// Traces Lambertian continuation rays through a linearly searched triangle span. Emission is
// accumulated at every hit and the required constant environment is accumulated on a miss. There
// is intentionally no next-event estimation, light sampling, MIS, or fallback backend. Its only
// scattering event is explicitly diffuse reflection. Surface emission remains visible when the
// diffuse limit is reached. Category counters are bound to PathState, so resumed paths cannot be
// reclassified. Russian roulette is explicitly enabled or disabled and uses the reserved
// per-bounce dimension after an accepted continuation. Secondary rays use [0, +infinity] while
// preserving time, mask, and medium.
[[nodiscard]] core::Result<BsdfOnlyPathResult> trace_bsdf_only(
    const Ray& initial_ray, const PathState& initial_state, const SampleStream& sample_stream,
    std::span<const BsdfOnlyTriangleSurface> surfaces, const BsdfOnlyEnvironment& environment,
    const PathDepthLimits& depth_limits, const RussianRoulettePolicy& roulette_policy);

[[nodiscard]] core::Result<ReferenceBsdfOnlyPathResult>
trace_bsdf_only(const ReferenceRay& initial_ray, const ReferencePathState& initial_state,
                const ReferenceSampleStream& sample_stream,
                std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces,
                const ReferenceBsdfOnlyEnvironment& environment,
                const PathDepthLimits& depth_limits,
                const ReferenceRussianRoulettePolicy& roulette_policy);

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
