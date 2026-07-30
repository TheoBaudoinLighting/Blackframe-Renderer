#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <cmath>
#include <numbers>
#include <string>

namespace blackframe::renderer {
namespace sampling_mappings_detail {

template <GeometryScalar Scalar>
[[nodiscard]] inline bool is_unit_interval(const Scalar value) noexcept {
    return std::isfinite(value) && value >= Scalar{0} && value < Scalar{1};
}

template <GeometryScalar Scalar>
[[nodiscard]] inline bool is_unit_square(const Point2T<Scalar> sample) noexcept {
    return is_unit_interval(sample.x) && is_unit_interval(sample.y);
}

[[nodiscard]] inline core::Error invalid_square_sample(const char* const mapping_name) {
    return {
        .code = core::StatusCode::invalid_argument,
        .message = std::string{mapping_name} +
                   " requires finite coordinates in the half-open unit square [0, 1).",
    };
}

template <GeometryScalar Scalar> struct ConcentricDiskMapping final {
    Point2T<Scalar> point;
    Scalar radius;
};

template <GeometryScalar Scalar>
[[nodiscard]] inline ConcentricDiskMapping<Scalar>
map_concentric_disk_unchecked(const Point2T<Scalar> sample) noexcept {
    const auto offset_x = Scalar{2} * sample.x - Scalar{1};
    const auto offset_y = Scalar{2} * sample.y - Scalar{1};
    if (offset_x == Scalar{0} && offset_y == Scalar{0}) {
        return {};
    }

    auto signed_radius = Scalar{};
    auto azimuth = Scalar{};
    constexpr auto quarter_pi = std::numbers::pi_v<Scalar> / Scalar{4};
    if (std::abs(offset_x) > std::abs(offset_y)) {
        signed_radius = offset_x;
        azimuth = quarter_pi * (offset_y / offset_x);
    } else {
        signed_radius = offset_y;
        azimuth = Scalar{2} * quarter_pi - quarter_pi * (offset_x / offset_y);
    }

    return {
        .point =
            {
                .x = signed_radius * std::cos(azimuth),
                .y = signed_radius * std::sin(azimuth),
            },
        .radius = std::abs(signed_radius),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] inline Scalar azimuth(const Scalar unit_sample) noexcept {
    return Scalar{2} * std::numbers::pi_v<Scalar> * unit_sample;
}

} // namespace sampling_mappings_detail

// Maps a canonical square sample uniformly onto the unit disk. The returned point lies in the
// local XY plane; uniform_disk_pdf() is its density with respect to area.
template <GeometryScalar Scalar>
[[nodiscard]] inline core::Result<Point2T<Scalar>>
map_concentric_disk(const Point2T<Scalar> sample) {
    if (!sampling_mappings_detail::is_unit_square(sample)) {
        return std::unexpected(
            sampling_mappings_detail::invalid_square_sample("Concentric disk mapping"));
    }
    return sampling_mappings_detail::map_concentric_disk_unchecked(sample).point;
}

// Maps a canonical square sample uniformly onto the unit sphere. The azimuth runs from +X toward
// +Y, and uniform_sphere_pdf() is measured with respect to solid angle.
template <GeometryScalar Scalar>
[[nodiscard]] inline core::Result<Vector3T<Scalar>>
map_uniform_sphere(const Point2T<Scalar> sample) {
    if (!sampling_mappings_detail::is_unit_square(sample)) {
        return std::unexpected(
            sampling_mappings_detail::invalid_square_sample("Uniform sphere mapping"));
    }

    const auto z = Scalar{1} - Scalar{2} * sample.x;
    const auto radial = Scalar{2} * std::sqrt(sample.x * (Scalar{1} - sample.x));
    if (radial == Scalar{0}) {
        return Vector3T<Scalar>{.x = Scalar{0}, .y = Scalar{0}, .z = z};
    }

    const auto phi = sampling_mappings_detail::azimuth(sample.y);
    return Vector3T<Scalar>{
        .x = radial * std::cos(phi),
        .y = radial * std::sin(phi),
        .z = z,
    };
}

// Maps a canonical square sample uniformly onto the +Z unit hemisphere.
// uniform_hemisphere_pdf() is measured with respect to solid angle.
template <GeometryScalar Scalar>
[[nodiscard]] inline core::Result<Vector3T<Scalar>>
map_uniform_hemisphere(const Point2T<Scalar> sample) {
    if (!sampling_mappings_detail::is_unit_square(sample)) {
        return std::unexpected(
            sampling_mappings_detail::invalid_square_sample("Uniform hemisphere mapping"));
    }

    const auto z = sample.x;
    const auto radial = std::sqrt((Scalar{1} - z) * (Scalar{1} + z));
    const auto phi = sampling_mappings_detail::azimuth(sample.y);
    return Vector3T<Scalar>{
        .x = radial * std::cos(phi),
        .y = radial * std::sin(phi),
        .z = z,
    };
}

// Lifts the concentric disk mapping onto the +Z unit hemisphere with a cosine distribution.
// cosine_hemisphere_pdf() is measured with respect to solid angle.
template <GeometryScalar Scalar>
[[nodiscard]] inline core::Result<Vector3T<Scalar>>
map_cosine_hemisphere(const Point2T<Scalar> sample) {
    if (!sampling_mappings_detail::is_unit_square(sample)) {
        return std::unexpected(
            sampling_mappings_detail::invalid_square_sample("Cosine hemisphere mapping"));
    }

    const auto disk = sampling_mappings_detail::map_concentric_disk_unchecked(sample);
    const auto z = std::sqrt((Scalar{1} - disk.radius) * (Scalar{1} + disk.radius));
    return Vector3T<Scalar>{
        .x = disk.point.x,
        .y = disk.point.y,
        .z = z,
    };
}

// Returns the on-support density for the unit disk; it is zero outside the disk.
template <GeometryScalar Scalar> [[nodiscard]] constexpr Scalar uniform_disk_pdf() noexcept {
    return Scalar{1} / std::numbers::pi_v<Scalar>;
}

// Returns the on-support density for the complete unit sphere.
template <GeometryScalar Scalar> [[nodiscard]] constexpr Scalar uniform_sphere_pdf() noexcept {
    return Scalar{1} / (Scalar{4} * std::numbers::pi_v<Scalar>);
}

// Returns the on-support density for the +Z unit hemisphere; it is zero below the horizon.
template <GeometryScalar Scalar> [[nodiscard]] constexpr Scalar uniform_hemisphere_pdf() noexcept {
    return Scalar{1} / (Scalar{2} * std::numbers::pi_v<Scalar>);
}

template <GeometryScalar Scalar>
[[nodiscard]] inline core::Result<Scalar> cosine_hemisphere_pdf(const Scalar cosine_theta) {
    if (!std::isfinite(cosine_theta) || cosine_theta < Scalar{-1} || cosine_theta > Scalar{1}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cosine hemisphere PDF requires a finite cosine in [-1, 1].",
        });
    }
    if (!(cosine_theta > Scalar{0})) {
        return Scalar{0};
    }
    return cosine_theta / std::numbers::pi_v<Scalar>;
}

} // namespace blackframe::renderer
