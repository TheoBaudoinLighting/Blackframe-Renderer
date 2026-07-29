#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <cmath>

namespace blackframe::renderer {

inline constexpr auto FixedDisplayPeak = ReferenceScalar{1};
inline constexpr auto FixedDisplayExposureMultiplier = ReferenceScalar{1};

namespace display_transform_detail {

inline constexpr auto srgb_linear_threshold = ReferenceScalar{0.0031308};
inline constexpr auto srgb_linear_scale = ReferenceScalar{12.92};
inline constexpr auto srgb_power_scale = ReferenceScalar{1.055};
inline constexpr auto srgb_power = ReferenceScalar{1} / ReferenceScalar{2.4};
inline constexpr auto srgb_power_offset = ReferenceScalar{0.055};

[[nodiscard]] inline ReferenceScalar encode_channel(const ReferenceScalar scene_linear) {
    const auto exposed = scene_linear * FixedDisplayExposureMultiplier;
    if (exposed <= ReferenceScalar{0}) {
        return ReferenceScalar{0};
    }
    if (exposed >= FixedDisplayPeak) {
        return FixedDisplayPeak;
    }
    if (exposed <= srgb_linear_threshold) {
        return srgb_linear_scale * exposed;
    }
    return srgb_power_scale * std::pow(exposed, srgb_power) - srgb_power_offset;
}

} // namespace display_transform_detail

// Applies the fixed preview display transform: zero-stop exposure, clipping to
// [0, 1], and the IEC 61966-2-1 sRGB transfer function. The result is
// display-referred RGB in double precision with an exact unit peak.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ReferenceLinearRGB>
apply_fixed_display_transform(const LinearRgbT<Scalar> scene_linear) {
    if (!std::isfinite(scene_linear.red) || !std::isfinite(scene_linear.green) ||
        !std::isfinite(scene_linear.blue)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The fixed display transform requires finite scene-linear RGB.",
        });
    }

    const auto display = ReferenceLinearRGB{
        .red = display_transform_detail::encode_channel(
            static_cast<ReferenceScalar>(scene_linear.red)),
        .green = display_transform_detail::encode_channel(
            static_cast<ReferenceScalar>(scene_linear.green)),
        .blue = display_transform_detail::encode_channel(
            static_cast<ReferenceScalar>(scene_linear.blue)),
    };
    if (!std::isfinite(display.red) || !std::isfinite(display.green) ||
        !std::isfinite(display.blue)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The fixed display transform produced non-finite RGB.",
        });
    }
    return display;
}

} // namespace blackframe::renderer
