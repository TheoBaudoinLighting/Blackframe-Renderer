#pragma once

#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace blackframe::renderer {

inline constexpr std::size_t TransportSpectrumSampleCount = 4;

template <typename Scalar>
concept SpectrumScalar =
    std::same_as<Scalar, TransportScalar> || std::same_as<Scalar, ReferenceScalar>;

template <std::size_t SampleCount>
concept TransportSpectrumExtent = SampleCount == TransportSpectrumSampleCount;

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
struct SampledSpectrumMask final {
    std::array<bool, SampleCount> lanes{};

    [[nodiscard]] constexpr bool operator==(const SampledSpectrumMask&) const noexcept = default;
};

template <std::size_t SampleCount, SpectrumScalar Scalar = TransportScalar>
    requires TransportSpectrumExtent<SampleCount>
struct SampledSpectrum final {
    std::array<Scalar, SampleCount> values{};

    [[nodiscard]] constexpr Scalar& operator[](const std::size_t index) noexcept {
        return values[index];
    }

    [[nodiscard]] constexpr Scalar operator[](const std::size_t index) const noexcept {
        return values[index];
    }

    [[nodiscard]] constexpr bool operator==(const SampledSpectrum&) const noexcept = default;
};

using TransportSpectrum = SampledSpectrum<TransportSpectrumSampleCount>;
using ReferenceSpectrum = SampledSpectrum<TransportSpectrumSampleCount, ReferenceScalar>;
using TransportSpectrumMask = SampledSpectrumMask<TransportSpectrumSampleCount>;

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator+(const SampledSpectrum<SampleCount, Scalar>& left,
          const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = left[index] + right[index];
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator-(const SampledSpectrum<SampleCount, Scalar>& left,
          const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = left[index] - right[index];
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator-(const SampledSpectrum<SampleCount, Scalar>& value) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = -value[index];
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator*(const SampledSpectrum<SampleCount, Scalar>& left,
          const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = left[index] * right[index];
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator*(const SampledSpectrum<SampleCount, Scalar>& value, const Scalar scale) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = value[index] * scale;
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator*(const Scalar scale, const SampledSpectrum<SampleCount, Scalar>& value) noexcept {
    return value * scale;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
operator/(const SampledSpectrum<SampleCount, Scalar>& value, const Scalar scale) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = value[index] / scale;
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar, typename Predicate>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
compare_spectrum(const SampledSpectrum<SampleCount, Scalar>& left,
                 const SampledSpectrum<SampleCount, Scalar>& right, Predicate predicate) noexcept {
    auto result = SampledSpectrumMask<SampleCount>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result.lanes[index] = predicate(left[index], right[index]);
    }
    return result;
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator<(const SampledSpectrum<SampleCount, Scalar>& left,
          const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    return compare_spectrum(left, right, [](const Scalar a, const Scalar b) { return a < b; });
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator<=(const SampledSpectrum<SampleCount, Scalar>& left,
           const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    return compare_spectrum(left, right, [](const Scalar a, const Scalar b) { return a <= b; });
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator>(const SampledSpectrum<SampleCount, Scalar>& left,
          const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    return compare_spectrum(left, right, [](const Scalar a, const Scalar b) { return a > b; });
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator>=(const SampledSpectrum<SampleCount, Scalar>& left,
           const SampledSpectrum<SampleCount, Scalar>& right) noexcept {
    return compare_spectrum(left, right, [](const Scalar a, const Scalar b) { return a >= b; });
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator&(const SampledSpectrumMask<SampleCount>& left,
          const SampledSpectrumMask<SampleCount>& right) noexcept {
    auto result = SampledSpectrumMask<SampleCount>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result.lanes[index] = left.lanes[index] && right.lanes[index];
    }
    return result;
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator|(const SampledSpectrumMask<SampleCount>& left,
          const SampledSpectrumMask<SampleCount>& right) noexcept {
    auto result = SampledSpectrumMask<SampleCount>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result.lanes[index] = left.lanes[index] || right.lanes[index];
    }
    return result;
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator^(const SampledSpectrumMask<SampleCount>& left,
          const SampledSpectrumMask<SampleCount>& right) noexcept {
    auto result = SampledSpectrumMask<SampleCount>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result.lanes[index] = left.lanes[index] != right.lanes[index];
    }
    return result;
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrumMask<SampleCount>
operator~(const SampledSpectrumMask<SampleCount>& value) noexcept {
    auto result = SampledSpectrumMask<SampleCount>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result.lanes[index] = !value.lanes[index];
    }
    return result;
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr bool any(const SampledSpectrumMask<SampleCount>& mask) noexcept {
    return std::ranges::any_of(mask.lanes, [](const bool lane) { return lane; });
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr bool all(const SampledSpectrumMask<SampleCount>& mask) noexcept {
    return std::ranges::all_of(mask.lanes, [](const bool lane) { return lane; });
}

template <std::size_t SampleCount>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr bool none(const SampledSpectrumMask<SampleCount>& mask) noexcept {
    return !any(mask);
}

template <std::size_t SampleCount, SpectrumScalar Scalar>
    requires TransportSpectrumExtent<SampleCount>
[[nodiscard]] constexpr SampledSpectrum<SampleCount, Scalar>
select(const SampledSpectrumMask<SampleCount>& mask,
       const SampledSpectrum<SampleCount, Scalar>& selected,
       const SampledSpectrum<SampleCount, Scalar>& rejected) noexcept {
    auto result = SampledSpectrum<SampleCount, Scalar>{};
    for (std::size_t index = 0; index < SampleCount; ++index) {
        result[index] = mask.lanes[index] ? selected[index] : rejected[index];
    }
    return result;
}

static_assert(std::is_standard_layout_v<TransportSpectrum>);
static_assert(std::is_trivially_copyable_v<TransportSpectrum>);
static_assert(sizeof(TransportSpectrum) == TransportSpectrumSampleCount * sizeof(TransportScalar));
static_assert(sizeof(ReferenceSpectrum) == TransportSpectrumSampleCount * sizeof(ReferenceScalar));

} // namespace blackframe::renderer
