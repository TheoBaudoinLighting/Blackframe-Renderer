#pragma once

#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace blackframe::renderer {

inline constexpr std::uint32_t MaximumClosureCount = 8U;
inline constexpr std::uint32_t ClosureParameterScalarCount = 10U;

// Codes are part of the renderer ABI. None is reserved for inactive inline slots; new closure
// models append values without changing the representation.
enum class ClosureKind : std::uint32_t {
    none = 0U,
    lambertian_reflection = 1U,
    rough_diffuse_reflection = 2U,
    rough_conductor_reflection = 3U,
    rough_dielectric = 4U,
    specular_reflection = 5U,
    specular_transmission = 6U,
};

[[nodiscard]] constexpr bool is_known_closure_kind(const ClosureKind kind) noexcept {
    switch (kind) {
    case ClosureKind::none:
    case ClosureKind::lambertian_reflection:
    case ClosureKind::rough_diffuse_reflection:
    case ClosureKind::rough_conductor_reflection:
    case ClosureKind::rough_dielectric:
    case ClosureKind::specular_reflection:
    case ClosureKind::specular_transmission:
        return true;
    }
    return false;
}

enum class ClosureAppendStatus : std::uint32_t {
    appended = 0U,
    invalid_payload = 1U,
    capacity_exhausted = 2U,
};

// Every slot is a fixed ABI record. Weight is the model's spectral coefficient. Lambertian
// reflection leaves all ten scalar parameters zero; rough diffuse stores normalized roughness in
// parameters[0]. Rough conductor stores spectral eta in parameters[0..3], spectral k in
// parameters[4..7], and GGX alphaX/alphaY in parameters[8..9]. Rough dielectric stores exterior
// eta, interior eta, and GGX alphaX/alphaY in parameters[0..3] and leaves parameters[4..9]
// reserved. Specular reflection leaves every parameter zero; specular transmission stores exterior
// and interior eta in parameters[0..1] and leaves parameters[2..9] reserved. Tangent-axis rotation
// belongs to the caller-supplied local closure frame and is not duplicated in an individual
// closure record.
template <SpectrumScalar Scalar> struct alignas(8) ClosureT final {
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

    ClosureKind kind{ClosureKind::none};
    ScatteringLobe lobes{ScatteringLobe::none};
    spectrum_type weight{};
    std::array<Scalar, ClosureParameterScalarCount> parameters{};
};

using Closure = ClosureT<TransportScalar>;
using ReferenceClosure = ClosureT<ReferenceScalar>;

namespace closure_set_detail {

template <SpectrumScalar Scalar> struct ClosureSetLayoutProbe;

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_reflectance(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& reflectance) noexcept {
    for (const auto value : reflectance.values) {
        if (!std::isfinite(value) || value < Scalar{0} || value > Scalar{1}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_roughness(const Scalar roughness) noexcept {
    return std::isfinite(roughness) && roughness >= Scalar{0} && roughness <= Scalar{1};
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_relative_eta(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_eta) noexcept {
    for (const auto value : relative_eta.values) {
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool
valid_relative_k(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_k) noexcept {
    for (const auto value : relative_k.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

} // namespace closure_set_detail

// The active prefix is stable and insertion ordered. Capacity exhaustion and invalid payloads are
// explicit statuses: no closure is merged, dropped, replaced, clamped, or allocated dynamically.
template <SpectrumScalar Scalar> class alignas(8) ClosureSetT final {
  public:
    using closure_type = ClosureT<Scalar>;
    using spectrum_type = typename closure_type::spectrum_type;

    [[nodiscard]] static constexpr std::uint32_t capacity() noexcept {
        return MaximumClosureCount;
    }

    [[nodiscard]] constexpr std::uint32_t size() const noexcept {
        return size_;
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0U;
    }
    [[nodiscard]] constexpr bool full() const noexcept {
        return size_ == MaximumClosureCount;
    }

    [[nodiscard]] constexpr std::span<const closure_type> closures() const noexcept {
        return {closures_.data(), static_cast<std::size_t>(size_)};
    }

    [[nodiscard]] ClosureAppendStatus
    append_lambertian_reflection(const spectrum_type reflectance) noexcept {
        if (!closure_set_detail::valid_reflectance(reflectance)) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        closures_[size_] = closure_type{
            .kind = ClosureKind::lambertian_reflection,
            .lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection,
            .weight = reflectance,
            .parameters = {},
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

    [[nodiscard]] ClosureAppendStatus
    append_rough_diffuse_reflection(const spectrum_type reflectance,
                                    const Scalar roughness) noexcept {
        if (!closure_set_detail::valid_reflectance(reflectance) ||
            !closure_set_detail::valid_roughness(roughness)) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        auto parameters = std::array<Scalar, ClosureParameterScalarCount>{};
        parameters[0] = roughness;
        closures_[size_] = closure_type{
            .kind = ClosureKind::rough_diffuse_reflection,
            .lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection,
            .weight = reflectance,
            .parameters = parameters,
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

    [[nodiscard]] ClosureAppendStatus
    append_rough_conductor_reflection(const spectrum_type coefficient,
                                      const spectrum_type relative_eta,
                                      const spectrum_type relative_k, const Scalar alpha) noexcept {
        return append_rough_conductor_reflection(coefficient, relative_eta, relative_k, alpha,
                                                 alpha);
    }

    [[nodiscard]] ClosureAppendStatus append_rough_conductor_reflection(
        const spectrum_type coefficient, const spectrum_type relative_eta,
        const spectrum_type relative_k, const Scalar alpha_x, const Scalar alpha_y) noexcept {
        if (!closure_set_detail::valid_reflectance(coefficient) ||
            !closure_set_detail::valid_relative_eta(relative_eta) ||
            !closure_set_detail::valid_relative_k(relative_k) ||
            !ggx_microfacet_detail::representable_alpha_pair(alpha_x, alpha_y)) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        auto parameters = std::array<Scalar, ClosureParameterScalarCount>{};
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            parameters[lane] = relative_eta[lane];
            parameters[TransportSpectrumSampleCount + lane] = relative_k[lane];
        }
        parameters[TransportSpectrumSampleCount * 2U] = alpha_x;
        parameters[TransportSpectrumSampleCount * 2U + 1U] = alpha_y;
        closures_[size_] = closure_type{
            .kind = ClosureKind::rough_conductor_reflection,
            .lobes = ScatteringLobe::glossy | ScatteringLobe::reflection,
            .weight = coefficient,
            .parameters = parameters,
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

    [[nodiscard]] ClosureAppendStatus append_rough_dielectric(const spectrum_type coefficient,
                                                              const Scalar exterior_eta,
                                                              const Scalar interior_eta,
                                                              const Scalar alpha) noexcept {
        return append_rough_dielectric(coefficient, exterior_eta, interior_eta, alpha, alpha);
    }

    [[nodiscard]] ClosureAppendStatus append_rough_dielectric(const spectrum_type coefficient,
                                                              const Scalar exterior_eta,
                                                              const Scalar interior_eta,
                                                              const Scalar alpha_x,
                                                              const Scalar alpha_y) noexcept {
        if (!closure_set_detail::valid_reflectance(coefficient) || !std::isfinite(exterior_eta) ||
            !(exterior_eta > Scalar{0}) || !std::isfinite(interior_eta) ||
            !(interior_eta > Scalar{0}) || exterior_eta == interior_eta ||
            !ggx_microfacet_detail::representable_alpha_pair(alpha_x, alpha_y)) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        auto parameters = std::array<Scalar, ClosureParameterScalarCount>{};
        parameters[0] = exterior_eta;
        parameters[1] = interior_eta;
        parameters[2] = alpha_x;
        parameters[3] = alpha_y;
        closures_[size_] = closure_type{
            .kind = ClosureKind::rough_dielectric,
            .lobes =
                ScatteringLobe::glossy | ScatteringLobe::reflection | ScatteringLobe::transmission,
            .weight = coefficient,
            .parameters = parameters,
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

    [[nodiscard]] ClosureAppendStatus
    append_specular_reflection(const spectrum_type reflectance) noexcept {
        if (!closure_set_detail::valid_reflectance(reflectance)) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        closures_[size_] = closure_type{
            .kind = ClosureKind::specular_reflection,
            .lobes = ScatteringLobe::specular | ScatteringLobe::reflection,
            .weight = reflectance,
            .parameters = {},
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

    [[nodiscard]] ClosureAppendStatus
    append_specular_transmission(const spectrum_type transmittance, const Scalar exterior_eta,
                                 const Scalar interior_eta) noexcept {
        if (!closure_set_detail::valid_reflectance(transmittance) || !std::isfinite(exterior_eta) ||
            !(exterior_eta > Scalar{0}) || !std::isfinite(interior_eta) ||
            !(interior_eta > Scalar{0})) {
            return ClosureAppendStatus::invalid_payload;
        }
        if (full()) {
            return ClosureAppendStatus::capacity_exhausted;
        }

        auto parameters = std::array<Scalar, ClosureParameterScalarCount>{};
        parameters[0] = exterior_eta;
        parameters[1] = interior_eta;
        closures_[size_] = closure_type{
            .kind = ClosureKind::specular_transmission,
            .lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
            .weight = transmittance,
            .parameters = parameters,
        };
        ++size_;
        return ClosureAppendStatus::appended;
    }

  private:
    friend struct closure_set_detail::ClosureSetLayoutProbe<Scalar>;

    std::uint32_t size_{};
    std::uint32_t reserved_{};
    std::array<closure_type, MaximumClosureCount> closures_{};
};

using ClosureSet = ClosureSetT<TransportScalar>;
using ReferenceClosureSet = ClosureSetT<ReferenceScalar>;

namespace closure_set_detail {

template <SpectrumScalar Scalar> struct ClosureSetLayoutProbe final {
    static constexpr std::size_t active_count_offset = offsetof(ClosureSetT<Scalar>, size_);
    static constexpr std::size_t reserved_offset = offsetof(ClosureSetT<Scalar>, reserved_);
    static constexpr std::size_t closures_offset = offsetof(ClosureSetT<Scalar>, closures_);
};

} // namespace closure_set_detail

static_assert(MaximumClosureCount == 8U);
static_assert(ClosureParameterScalarCount == 10U);
static_assert(sizeof(ClosureKind) == sizeof(std::uint32_t));
static_assert(sizeof(ClosureAppendStatus) == sizeof(std::uint32_t));
static_assert(static_cast<std::uint32_t>(ClosureKind::none) == 0U);
static_assert(static_cast<std::uint32_t>(ClosureKind::lambertian_reflection) == 1U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_diffuse_reflection) == 2U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_conductor_reflection) == 3U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_dielectric) == 4U);
static_assert(static_cast<std::uint32_t>(ClosureKind::specular_reflection) == 5U);
static_assert(static_cast<std::uint32_t>(ClosureKind::specular_transmission) == 6U);
static_assert(static_cast<std::uint32_t>(ClosureAppendStatus::appended) == 0U);
static_assert(static_cast<std::uint32_t>(ClosureAppendStatus::invalid_payload) == 1U);
static_assert(static_cast<std::uint32_t>(ClosureAppendStatus::capacity_exhausted) == 2U);

static_assert(std::is_standard_layout_v<Closure>);
static_assert(std::is_trivially_copyable_v<Closure>);
static_assert(std::is_trivially_destructible_v<Closure>);
static_assert(sizeof(Closure) == 64U);
static_assert(alignof(Closure) == 8U);
static_assert(offsetof(Closure, kind) == 0U);
static_assert(offsetof(Closure, lobes) == 4U);
static_assert(offsetof(Closure, weight) == 8U);
static_assert(offsetof(Closure, parameters) == 24U);
static_assert(std::is_standard_layout_v<ReferenceClosure>);
static_assert(std::is_trivially_copyable_v<ReferenceClosure>);
static_assert(std::is_trivially_destructible_v<ReferenceClosure>);
static_assert(sizeof(ReferenceClosure) == 120U);
static_assert(alignof(ReferenceClosure) == 8U);
static_assert(offsetof(ReferenceClosure, kind) == 0U);
static_assert(offsetof(ReferenceClosure, lobes) == 4U);
static_assert(offsetof(ReferenceClosure, weight) == 8U);
static_assert(offsetof(ReferenceClosure, parameters) == 40U);

static_assert(std::is_standard_layout_v<ClosureSet>);
static_assert(std::is_trivially_copyable_v<ClosureSet>);
static_assert(std::is_trivially_destructible_v<ClosureSet>);
static_assert(sizeof(ClosureSet) == 520U);
static_assert(alignof(ClosureSet) == 8U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::active_count_offset ==
              0U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::reserved_offset == 4U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<TransportScalar>::closures_offset == 8U);
static_assert(std::is_standard_layout_v<ReferenceClosureSet>);
static_assert(std::is_trivially_copyable_v<ReferenceClosureSet>);
static_assert(std::is_trivially_destructible_v<ReferenceClosureSet>);
static_assert(sizeof(ReferenceClosureSet) == 968U);
static_assert(alignof(ReferenceClosureSet) == 8U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::active_count_offset ==
              0U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::reserved_offset == 4U);
static_assert(closure_set_detail::ClosureSetLayoutProbe<ReferenceScalar>::closures_offset == 8U);

} // namespace blackframe::renderer
