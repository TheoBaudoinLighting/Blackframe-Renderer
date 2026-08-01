#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cmath>
#include <limits>

namespace blackframe::renderer {
namespace shading_normal_correction_detail {

[[nodiscard]] inline core::Error invalid_shading_normal_correction(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] bool finite_unit_direction(const Vector3T<Scalar> direction) noexcept {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return false;
    }
    const auto squared_length = std::fma(
        direction.x, direction.x, std::fma(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool finite_unit_normal(const Normal3T<Scalar> normal) noexcept {
    if (!std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z)) {
        return false;
    }
    const auto squared_length =
        std::fma(normal.x, normal.x, std::fma(normal.y, normal.y, normal.z * normal.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> checked_importance_ratio(const Scalar outgoing_shading_cosine,
                                                            const Scalar incoming_geometric_cosine,
                                                            const Scalar outgoing_geometric_cosine,
                                                            const Scalar incoming_shading_cosine) {
    auto outgoing_shading_exponent = 0;
    auto incoming_geometric_exponent = 0;
    auto outgoing_geometric_exponent = 0;
    auto incoming_shading_exponent = 0;
    const auto numerator =
        std::frexp(std::abs(outgoing_shading_cosine), &outgoing_shading_exponent) *
        std::frexp(std::abs(incoming_geometric_cosine), &incoming_geometric_exponent);
    const auto denominator =
        std::frexp(std::abs(outgoing_geometric_cosine), &outgoing_geometric_exponent) *
        std::frexp(std::abs(incoming_shading_cosine), &incoming_shading_exponent);
    const auto numerator_exponent = outgoing_shading_exponent + incoming_geometric_exponent;
    const auto denominator_exponent = outgoing_geometric_exponent + incoming_shading_exponent;
    const auto ratio =
        std::scalbn(numerator / denominator, numerator_exponent - denominator_exponent);
    if (!std::isfinite(ratio) || !(ratio > Scalar{0})) {
        return std::unexpected(invalid_shading_normal_correction(
            "The shading-normal importance correction is not representable."));
    }
    return ratio;
}

} // namespace shading_normal_correction_detail

// Ng defines geometric sidedness, ray offsets, visibility, and geometric Jacobians. Ns defines the
// local closure frame, including its cosine and directional PDF. Under this convention Veach's
// adjoint correction is exactly one for radiance transport and
//
//   |wo . Ns| |wi . Ng| / (|wo . Ng| |wi . Ns|)
//
// for importance transport. Directions must lie on a side on which Ng and Ns agree. Exact tangent
// or conflicting directions have zero support; they are never face-forwarded or epsilon-clamped.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
shading_normal_correction(const Normal3T<Scalar> geometric_normal,
                          const Normal3T<Scalar> shading_normal,
                          const Vector3T<Scalar> outgoing_world,
                          const Vector3T<Scalar> incoming_world, const TransportMode mode) {
    if (!is_known_transport_mode(mode)) {
        return std::unexpected(shading_normal_correction_detail::invalid_shading_normal_correction(
            "A shading-normal correction requires a known transport mode."));
    }
    if (!shading_normal_correction_detail::finite_unit_normal(geometric_normal) ||
        !shading_normal_correction_detail::finite_unit_normal(shading_normal)) {
        return std::unexpected(shading_normal_correction_detail::invalid_shading_normal_correction(
            "Shading-normal correction normals must be finite unit normals."));
    }
    if (!shading_normal_correction_detail::finite_unit_direction(outgoing_world) ||
        !shading_normal_correction_detail::finite_unit_direction(incoming_world)) {
        return std::unexpected(shading_normal_correction_detail::invalid_shading_normal_correction(
            "Shading-normal correction directions must be finite unit vectors."));
    }
    if (!(dot(geometric_normal, shading_normal) > Scalar{0})) {
        return std::unexpected(shading_normal_correction_detail::invalid_shading_normal_correction(
            "Geometric and shading normals must share an open hemisphere."));
    }

    const auto outgoing_geometric_cosine = dot(geometric_normal, outgoing_world);
    const auto outgoing_shading_cosine = dot(shading_normal, outgoing_world);
    const auto incoming_geometric_cosine = dot(geometric_normal, incoming_world);
    const auto incoming_shading_cosine = dot(shading_normal, incoming_world);
    if (outgoing_geometric_cosine == Scalar{0} || outgoing_shading_cosine == Scalar{0} ||
        incoming_geometric_cosine == Scalar{0} || incoming_shading_cosine == Scalar{0} ||
        std::signbit(outgoing_geometric_cosine) != std::signbit(outgoing_shading_cosine) ||
        std::signbit(incoming_geometric_cosine) != std::signbit(incoming_shading_cosine)) {
        return Scalar{0};
    }
    if (mode == TransportMode::radiance || geometric_normal == shading_normal) {
        return Scalar{1};
    }
    return shading_normal_correction_detail::checked_importance_ratio(
        outgoing_shading_cosine, incoming_geometric_cosine, outgoing_geometric_cosine,
        incoming_shading_cosine);
}

} // namespace blackframe::renderer
