#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/DisplayTransform.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::renderer {

struct DisplayPsnrResult final {
    // Positive infinity represents an exact display-referred match.
    ReferenceScalar psnr{};
    FilmCrop crop{};

    // Row-major scalar heatmap over crop. Each value is the maximum absolute
    // display-referred difference among the pixel's R, G, and B components.
    std::vector<ReferenceScalar> difference_heatmap;

    [[nodiscard]] bool operator==(const DisplayPsnrResult&) const noexcept = default;
};

namespace display_psnr_detail {

[[nodiscard]] inline core::Error make_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
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
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "Display PSNR requires identical full film extents."));
    }
    if (evaluated.crop() != reference.crop()) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "Display PSNR requires identical active film crops."));
    }
    return {};
}

class ScaledSquareSum final {
  public:
    [[nodiscard]] core::Status add(const ReferenceScalar value) {
        if (!std::isfinite(value) || value < ReferenceScalar{0} || value > FixedDisplayPeak) {
            return std::unexpected(make_error(
                core::StatusCode::invalid_argument,
                "Display PSNR requires finite component differences within the unit peak."));
        }
        if (value == ReferenceScalar{0}) {
            return {};
        }
        if (scale_ < value) {
            const auto ratio = scale_ / value;
            scaled_sum_ = ReferenceScalar{1} + scaled_sum_ * ratio * ratio;
            scale_ = value;
        } else {
            const auto ratio = value / scale_;
            scaled_sum_ += ratio * ratio;
        }
        if (!std::isfinite(scaled_sum_)) {
            return std::unexpected(
                make_error(core::StatusCode::invalid_argument,
                           "Display PSNR square accumulation overflowed double precision."));
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

} // namespace display_psnr_detail

// Applies the same fixed display transform as PNG previews, computes
// component-wise RGB MSE, then PSNR = 10*log10(1/MSE) with a fixed peak of one.
// Exact matches return positive infinity. Inputs must have identical, fully
// resolved crops; no alternate transform or epsilon is selected.
template <AccumulationPrecision EvaluatedPrecision, AccumulationPrecision ReferencePrecision>
[[nodiscard]] core::Result<DisplayPsnrResult>
compute_display_psnr(const FilmT<EvaluatedPrecision>& evaluated,
                     const FilmT<ReferencePrecision>& reference) {
    const auto domain_status = display_psnr_detail::validate_domains(evaluated, reference);
    if (!domain_status.has_value()) {
        return std::unexpected(domain_status.error());
    }

    constexpr auto rgb_component_count = std::size_t{3};
    if (evaluated.pixel_count() > std::numeric_limits<std::size_t>::max() / rgb_component_count) {
        return std::unexpected(display_psnr_detail::make_error(
            core::StatusCode::resource_exhausted,
            "The display PSNR component count is not representable."));
    }

    try {
        auto result = DisplayPsnrResult{
            .psnr = ReferenceScalar{},
            .crop = evaluated.crop(),
            .difference_heatmap = {},
        };
        result.difference_heatmap.reserve(evaluated.pixel_count());
        auto squared_error_sum = display_psnr_detail::ScaledSquareSum{};
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

                const auto evaluated_display = apply_fixed_display_transform(*evaluated_pixel);
                if (!evaluated_display.has_value()) {
                    return std::unexpected(evaluated_display.error());
                }
                const auto reference_display = apply_fixed_display_transform(*reference_pixel);
                if (!reference_display.has_value()) {
                    return std::unexpected(reference_display.error());
                }

                const auto differences = std::array{
                    std::abs(evaluated_display->red - reference_display->red),
                    std::abs(evaluated_display->green - reference_display->green),
                    std::abs(evaluated_display->blue - reference_display->blue),
                };
                auto pixel_maximum = ReferenceScalar{};
                for (const auto difference : differences) {
                    const auto square_status = squared_error_sum.add(difference);
                    if (!square_status.has_value()) {
                        return std::unexpected(square_status.error());
                    }
                    pixel_maximum = std::max(pixel_maximum, difference);
                }
                result.difference_heatmap.push_back(pixel_maximum);
            }
        }

        const auto component_count = evaluated.pixel_count() * rgb_component_count;
        result.psnr = squared_error_sum.scale() == ReferenceScalar{0}
                          ? std::numeric_limits<ReferenceScalar>::infinity()
                          : ReferenceScalar{20} * std::log10(FixedDisplayPeak) -
                                ReferenceScalar{20} * std::log10(squared_error_sum.scale()) +
                                ReferenceScalar{10} *
                                    std::log10(static_cast<ReferenceScalar>(component_count) /
                                               squared_error_sum.scaled_sum());
        if (std::isnan(result.psnr) ||
            (squared_error_sum.scale() != ReferenceScalar{0} && !std::isfinite(result.psnr))) {
            return std::unexpected(display_psnr_detail::make_error(
                core::StatusCode::invalid_argument, "Display PSNR produced an undefined result."));
        }
        return result;
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            display_psnr_detail::make_error(core::StatusCode::resource_exhausted,
                                            "The display difference heatmap cannot be allocated."));
    } catch (const std::length_error&) {
        return std::unexpected(display_psnr_detail::make_error(
            core::StatusCode::resource_exhausted,
            "The display difference heatmap size is not representable."));
    }
}

} // namespace blackframe::renderer
