#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace blackframe::renderer {

struct HdrMetrics final {
    ReferenceScalar relative_mse{};
    ReferenceScalar luminance_smape{};

    [[nodiscard]] constexpr bool operator==(const HdrMetrics&) const noexcept = default;
};

namespace hdr_metrics_detail {

[[nodiscard]] inline core::Error invalid_hdr_metrics_input(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

template <AccumulationPrecision EvaluatedPrecision, AccumulationPrecision ReferencePrecision>
[[nodiscard]] core::Status validate_domains(const FilmT<EvaluatedPrecision>& evaluated,
                                            const FilmT<ReferencePrecision>& reference) {
    const auto evaluated_extent = evaluated.extent();
    const auto reference_extent = reference.extent();
    if (evaluated_extent.width != reference_extent.width ||
        evaluated_extent.height != reference_extent.height) {
        return std::unexpected(
            invalid_hdr_metrics_input("HDR metrics require identical full film extents."));
    }
    if (evaluated.crop() != reference.crop()) {
        return std::unexpected(
            invalid_hdr_metrics_input("HDR metrics require identical active film crops."));
    }
    return {};
}

class ScaledSquareSum final {
  public:
    [[nodiscard]] core::Status add(const ReferenceScalar value) {
        if (!std::isfinite(value)) {
            return std::unexpected(invalid_hdr_metrics_input(
                "HDR metric square accumulation requires a finite component."));
        }

        const auto magnitude = std::abs(value);
        if (magnitude == ReferenceScalar{0}) {
            return {};
        }
        if (scale_ < magnitude) {
            const auto ratio = scale_ / magnitude;
            scaled_sum_ = ReferenceScalar{1} + scaled_sum_ * ratio * ratio;
            scale_ = magnitude;
        } else {
            const auto ratio = magnitude / scale_;
            scaled_sum_ += ratio * ratio;
        }
        if (!std::isfinite(scaled_sum_)) {
            return std::unexpected(invalid_hdr_metrics_input(
                "HDR metric square accumulation overflowed double precision."));
        }
        return {};
    }

    [[nodiscard]] ReferenceScalar scale() const noexcept {
        return scale_;
    }

    [[nodiscard]] ReferenceScalar scaled_sum() const noexcept {
        return scaled_sum_;
    }

  private:
    ReferenceScalar scale_{};
    ReferenceScalar scaled_sum_{};
};

[[nodiscard]] inline core::Result<ReferenceScalar>
square_sum_ratio(const ScaledSquareSum& numerator, const ScaledSquareSum& denominator) {
    if (denominator.scale() == ReferenceScalar{0}) {
        return std::unexpected(invalid_hdr_metrics_input(
            "Relative MSE is undefined for a reference with zero RGB energy."));
    }
    if (numerator.scale() == ReferenceScalar{0}) {
        return ReferenceScalar{0};
    }

    const auto scale_ratio = numerator.scale() / denominator.scale();
    if (!std::isfinite(scale_ratio) || scale_ratio == ReferenceScalar{0}) {
        return std::unexpected(invalid_hdr_metrics_input(
            "The relative MSE scale ratio is not representable in double precision."));
    }
    const auto scale_ratio_squared = scale_ratio * scale_ratio;
    const auto relative_mse =
        scale_ratio_squared * (numerator.scaled_sum() / denominator.scaled_sum());
    if (!std::isfinite(relative_mse) || relative_mse == ReferenceScalar{0}) {
        return std::unexpected(invalid_hdr_metrics_input(
            "The relative MSE result is not representable in double precision."));
    }
    return relative_mse;
}

template <typename Color>
[[nodiscard]] core::Result<ReferenceScalar> linear_luminance(const Color color) {
    const auto luminance = ReferenceScalar{0.2126729} * static_cast<ReferenceScalar>(color.red) +
                           ReferenceScalar{0.7151522} * static_cast<ReferenceScalar>(color.green) +
                           ReferenceScalar{0.0721750} * static_cast<ReferenceScalar>(color.blue);
    if (!std::isfinite(luminance)) {
        return std::unexpected(invalid_hdr_metrics_input(
            "HDR metric luminance conversion produced a non-finite value."));
    }
    return luminance;
}

[[nodiscard]] inline ReferenceScalar luminance_smape_term(const ReferenceScalar evaluated,
                                                          const ReferenceScalar reference) {
    const auto scale = std::max(std::abs(evaluated), std::abs(reference));
    if (scale == ReferenceScalar{0}) {
        return ReferenceScalar{0};
    }
    const auto normalized_evaluated = evaluated / scale;
    const auto normalized_reference = reference / scale;
    return ReferenceScalar{2} * std::abs(normalized_evaluated - normalized_reference) /
           (std::abs(normalized_evaluated) + std::abs(normalized_reference));
}

} // namespace hdr_metrics_detail

// Relative MSE is the RGB component square-error sum divided by the reference
// RGB square sum. Luminance SMAPE is the pixel mean of
// 2*|Y_evaluated-Y_reference|/(|Y_evaluated|+|Y_reference|), with an exact
// zero contribution when both luminances are zero. Both metrics are unitless
// and invariant under a shared positive scale.
template <AccumulationPrecision EvaluatedPrecision, AccumulationPrecision ReferencePrecision>
[[nodiscard]] core::Result<HdrMetrics>
compute_hdr_metrics(const FilmT<EvaluatedPrecision>& evaluated,
                    const FilmT<ReferencePrecision>& reference) {
    const auto domain_status = hdr_metrics_detail::validate_domains(evaluated, reference);
    if (!domain_status.has_value()) {
        return std::unexpected(domain_status.error());
    }

    auto squared_error_sum = hdr_metrics_detail::ScaledSquareSum{};
    auto squared_reference_sum = hdr_metrics_detail::ScaledSquareSum{};
    auto luminance_smape_sum = ReferenceScalar{};
    const auto crop = evaluated.crop();
    for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
            const auto evaluated_pixel = evaluated.resolved_pixel(x, y);
            if (!evaluated_pixel.has_value()) {
                return std::unexpected(evaluated_pixel.error());
            }
            const auto reference_pixel = reference.resolved_pixel(x, y);
            if (!reference_pixel.has_value()) {
                return std::unexpected(reference_pixel.error());
            }

            const auto evaluated_components = std::array{
                static_cast<ReferenceScalar>(evaluated_pixel->red),
                static_cast<ReferenceScalar>(evaluated_pixel->green),
                static_cast<ReferenceScalar>(evaluated_pixel->blue),
            };
            const auto reference_components = std::array{
                static_cast<ReferenceScalar>(reference_pixel->red),
                static_cast<ReferenceScalar>(reference_pixel->green),
                static_cast<ReferenceScalar>(reference_pixel->blue),
            };
            for (auto component = std::size_t{}; component < evaluated_components.size();
                 ++component) {
                const auto error =
                    evaluated_components[component] - reference_components[component];
                const auto error_status = squared_error_sum.add(error);
                if (!error_status.has_value()) {
                    return std::unexpected(error_status.error());
                }
                const auto reference_status =
                    squared_reference_sum.add(reference_components[component]);
                if (!reference_status.has_value()) {
                    return std::unexpected(reference_status.error());
                }
            }

            const auto evaluated_luminance = hdr_metrics_detail::linear_luminance(*evaluated_pixel);
            if (!evaluated_luminance.has_value()) {
                return std::unexpected(evaluated_luminance.error());
            }
            const auto reference_luminance = hdr_metrics_detail::linear_luminance(*reference_pixel);
            if (!reference_luminance.has_value()) {
                return std::unexpected(reference_luminance.error());
            }
            luminance_smape_sum += hdr_metrics_detail::luminance_smape_term(*evaluated_luminance,
                                                                            *reference_luminance);
            if (!std::isfinite(luminance_smape_sum)) {
                return std::unexpected(hdr_metrics_detail::invalid_hdr_metrics_input(
                    "Luminance SMAPE accumulation overflowed double precision."));
            }
        }
    }

    const auto relative_mse =
        hdr_metrics_detail::square_sum_ratio(squared_error_sum, squared_reference_sum);
    if (!relative_mse.has_value()) {
        return std::unexpected(relative_mse.error());
    }
    const auto luminance_smape =
        luminance_smape_sum / static_cast<ReferenceScalar>(evaluated.pixel_count());
    if (!std::isfinite(luminance_smape)) {
        return std::unexpected(hdr_metrics_detail::invalid_hdr_metrics_input(
            "Luminance SMAPE normalization produced a non-finite result."));
    }

    return HdrMetrics{
        .relative_mse = *relative_mse,
        .luminance_smape = luminance_smape,
    };
}

} // namespace blackframe::renderer
