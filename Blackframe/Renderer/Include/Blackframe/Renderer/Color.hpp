#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <cmath>
#include <type_traits>

namespace blackframe::renderer {

template <SpectrumScalar Scalar> struct XyzT final {
    Scalar x{};
    Scalar y{};
    Scalar z{};

    [[nodiscard]] constexpr bool operator==(const XyzT&) const noexcept = default;
};

template <SpectrumScalar Scalar> struct LinearRgbT final {
    Scalar red{};
    Scalar green{};
    Scalar blue{};

    [[nodiscard]] constexpr bool operator==(const LinearRgbT&) const noexcept = default;
};

using XYZ = XyzT<TransportScalar>;
using ReferenceXYZ = XyzT<ReferenceScalar>;
using LinearRGB = LinearRgbT<TransportScalar>;
using ReferenceLinearRGB = LinearRgbT<ReferenceScalar>;

namespace color_detail {

template <typename Color> [[nodiscard]] bool finite(const Color color) noexcept {
    if constexpr (requires { color.x; }) {
        return std::isfinite(color.x) && std::isfinite(color.y) && std::isfinite(color.z);
    } else {
        return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue);
    }
}

template <typename Color> [[nodiscard]] core::Result<Color> conversion_error() {
    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "A color conversion requires finite components and a finite result.",
    });
}

template <typename Color> [[nodiscard]] core::Result<Color> finite_result(const Color color) {
    if (!finite(color)) {
        return conversion_error<Color>();
    }
    return color;
}

} // namespace color_detail

// CIE XYZ uses the 1931 2-degree observer. Linear RGB uses the D65-referred sRGB primaries without
// an opto-electronic transfer function or chromatic adaptation. Out-of-gamut values remain signed
// and unclamped.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<XyzT<Scalar>> linear_rgb_to_xyz(const LinearRgbT<Scalar> rgb) {
    if (!color_detail::finite(rgb)) {
        return color_detail::conversion_error<XyzT<Scalar>>();
    }

    return color_detail::finite_result(XyzT<Scalar>{
        .x = Scalar{0.4124564} * rgb.red + Scalar{0.3575761} * rgb.green +
             Scalar{0.1804375} * rgb.blue,
        .y = Scalar{0.2126729} * rgb.red + Scalar{0.7151522} * rgb.green +
             Scalar{0.0721750} * rgb.blue,
        .z = Scalar{0.0193339} * rgb.red + Scalar{0.1191920} * rgb.green +
             Scalar{0.9503041} * rgb.blue,
    });
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LinearRgbT<Scalar>> xyz_to_linear_rgb(const XyzT<Scalar> xyz) {
    if (!color_detail::finite(xyz)) {
        return color_detail::conversion_error<LinearRgbT<Scalar>>();
    }

    return color_detail::finite_result(LinearRgbT<Scalar>{
        .red = Scalar{3.2404542} * xyz.x - Scalar{1.5371385} * xyz.y - Scalar{0.4985314} * xyz.z,
        .green = -Scalar{0.9692660} * xyz.x + Scalar{1.8760108} * xyz.y + Scalar{0.0415560} * xyz.z,
        .blue = Scalar{0.0556434} * xyz.x - Scalar{0.2040259} * xyz.y + Scalar{1.0572252} * xyz.z,
    });
}

static_assert(!std::is_same_v<XYZ, LinearRGB>);
static_assert(std::is_standard_layout_v<XYZ>);
static_assert(std::is_trivially_copyable_v<XYZ>);
static_assert(std::is_standard_layout_v<LinearRGB>);
static_assert(std::is_trivially_copyable_v<LinearRGB>);
static_assert(sizeof(XYZ) == 3 * sizeof(TransportScalar));
static_assert(sizeof(LinearRGB) == 3 * sizeof(TransportScalar));

} // namespace blackframe::renderer
