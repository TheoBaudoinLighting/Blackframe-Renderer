#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

namespace blackframe::renderer {

enum class DebugIdentifierKind : std::uint8_t {
    instance = 0,
    geometry = 1,
    primitive = 2,
    material = 3,
};

namespace debug_integrator_detail {

[[nodiscard]] inline core::Error debug_integrator_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr LinearRgbT<Scalar>
encode_identifier(const std::uint32_t identifier) noexcept {
    constexpr auto low_mask = std::uint32_t{0x7FF};
    constexpr auto high_mask = std::uint32_t{0x3FF};
    const auto low = identifier & low_mask;
    const auto middle = (identifier >> 11U) & low_mask;
    const auto high = (identifier >> 22U) & high_mask;
    return {
        .red = static_cast<Scalar>(low + 1U) / Scalar{2048},
        .green = static_cast<Scalar>(middle + 1U) / Scalar{2048},
        .blue = static_cast<Scalar>(high + 1U) / Scalar{1024},
    };
}

} // namespace debug_integrator_detail

// Encodes the world-space geometric normal. It is never face-forwarded and Ns
// is deliberately not substituted for Ng.
template <GeometryScalar Scalar>
[[nodiscard]] constexpr LinearRgbT<Scalar>
debug_normal_color(const SurfaceInteractionT<Scalar>& interaction) noexcept {
    const auto& normal = interaction.geometric_normal();
    return {
        .red = (normal.x + Scalar{1}) / Scalar{2},
        .green = (normal.y + Scalar{1}) / Scalar{2},
        .blue = (normal.z + Scalar{1}) / Scalar{2},
    };
}

// Depth is the unmodified primary-hit ray parameter. No implicit camera-space
// conversion, range normalization, logarithm, inversion, or clipping is used.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<LinearRgbT<Scalar>> debug_depth_color(const Scalar parameter) {
    if (!std::isfinite(parameter) || parameter < Scalar{0}) {
        return std::unexpected(debug_integrator_detail::debug_integrator_error(
            "Debug depth requires a finite non-negative ray parameter."));
    }
    return LinearRgbT<Scalar>{
        .red = parameter,
        .green = parameter,
        .blue = parameter,
    };
}

// UV coordinates remain signed and unbounded. Wrapping and display clipping
// belong to later, explicitly selected output stages.
template <GeometryScalar Scalar>
[[nodiscard]] constexpr LinearRgbT<Scalar>
debug_uv_color(const SurfaceInteractionT<Scalar>& interaction) noexcept {
    return {
        .red = interaction.uv().x,
        .green = interaction.uv().y,
        .blue = Scalar{0},
    };
}

// Triangle barycentrics preserve vertex order and are never clamped or
// renormalized. Invalid standalone coordinates fail before producing a color.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<LinearRgbT<Scalar>>
debug_barycentric_color(const TriangleBarycentricsT<Scalar> barycentrics) {
    if (!std::isfinite(barycentrics.vertex0) || !std::isfinite(barycentrics.vertex1) ||
        !std::isfinite(barycentrics.vertex2) || barycentrics.vertex0 < Scalar{0} ||
        barycentrics.vertex0 > Scalar{1} || barycentrics.vertex1 < Scalar{0} ||
        barycentrics.vertex1 > Scalar{1} || barycentrics.vertex2 < Scalar{0} ||
        barycentrics.vertex2 > Scalar{1}) {
        return std::unexpected(debug_integrator_detail::debug_integrator_error(
            "Debug barycentrics must be finite coordinates inside [0, 1]."));
    }

    const auto sum = barycentrics.vertex0 + barycentrics.vertex1 + barycentrics.vertex2;
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    if (!std::isfinite(sum) || std::abs(sum - Scalar{1}) > tolerance) {
        return std::unexpected(debug_integrator_detail::debug_integrator_error(
            "Debug barycentrics must sum to one within transport tolerance."));
    }
    return LinearRgbT<Scalar>{
        .red = barycentrics.vertex0,
        .green = barycentrics.vertex1,
        .blue = barycentrics.vertex2,
    };
}

// The caller must select one identifier category. Its complete 32 bits are
// split injectively into 11, 11, and 10-bit scene-linear RGB components. Each
// chunk is shifted before power-of-two scaling so no valid ID maps to the black
// miss color. No platform-dependent hash is involved.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<LinearRgbT<Scalar>>
debug_identifier_color(const SurfaceIdentifiers identifiers, const DebugIdentifierKind kind) {
    auto identifier = std::uint32_t{};
    switch (kind) {
    case DebugIdentifierKind::instance:
        identifier = identifiers.instance.value;
        break;
    case DebugIdentifierKind::geometry:
        identifier = identifiers.geometry.value;
        break;
    case DebugIdentifierKind::primitive:
        identifier = identifiers.primitive.value;
        break;
    case DebugIdentifierKind::material:
        identifier = identifiers.material.value;
        break;
    default:
        return std::unexpected(debug_integrator_detail::debug_integrator_error(
            "Unsupported debug identifier category."));
    }
    return debug_integrator_detail::encode_identifier<Scalar>(identifier);
}

} // namespace blackframe::renderer
