#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::renderer {

template <AccumulationPrecision Precision> struct FilmPixelT final {
    using Scalar = AccumulationScalar<Precision>;
    using Color = LinearRgbT<Scalar>;

    Color weighted_sum{};
    Scalar weight_sum{};
    std::uint64_t sample_count{};

    [[nodiscard]] constexpr bool operator==(const FilmPixelT&) const noexcept = default;
};

template <AccumulationPrecision Precision> class FilmT final {
  public:
    using Scalar = AccumulationScalar<Precision>;
    using Color = LinearRgbT<Scalar>;
    using Pixel = FilmPixelT<Precision>;

    [[nodiscard]] static core::Result<FilmT> create(const RenderExtent extent) {
        const auto extent_status = validate_render_extent(extent);
        if (!extent_status.has_value()) {
            return std::unexpected(extent_status.error());
        }

        const auto pixel_count =
            static_cast<std::size_t>(extent.width) * static_cast<std::size_t>(extent.height);
        try {
            return FilmT{extent, std::vector<Pixel>(pixel_count)};
        } catch (const std::bad_alloc&) {
            return allocation_error();
        } catch (const std::length_error&) {
            return allocation_error();
        }
    }

    [[nodiscard]] constexpr RenderExtent extent() const noexcept {
        return extent_;
    }

    [[nodiscard]] std::size_t pixel_count() const noexcept {
        return pixels_.size();
    }

    [[nodiscard]] core::Result<Pixel> pixel(const std::uint32_t x, const std::uint32_t y) const {
        const auto index = pixel_index(x, y);
        if (!index.has_value()) {
            return std::unexpected(index.error());
        }
        return pixels_[*index];
    }

    [[nodiscard]] core::Status add_sample(const std::uint32_t x, const std::uint32_t y,
                                          const Color sample, const Scalar weight) {
        const auto index = pixel_index(x, y);
        if (!index.has_value()) {
            return std::unexpected(index.error());
        }
        if (!finite(sample) || !std::isfinite(weight)) {
            return std::unexpected(invalid_sample_error(
                "Film samples and their reconstruction weights must be finite."));
        }

        const auto& current = pixels_[*index];
        if (current.sample_count == std::numeric_limits<std::uint64_t>::max()) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::resource_exhausted,
                .message = "The film sample counter cannot represent another sample.",
            });
        }

        const auto candidate = Pixel{
            .weighted_sum =
                Color{
                    .red = current.weighted_sum.red + sample.red * weight,
                    .green = current.weighted_sum.green + sample.green * weight,
                    .blue = current.weighted_sum.blue + sample.blue * weight,
                },
            .weight_sum = current.weight_sum + weight,
            .sample_count = current.sample_count + 1,
        };
        if (!finite(candidate.weighted_sum) || !std::isfinite(candidate.weight_sum)) {
            return std::unexpected(
                invalid_sample_error("Film accumulation produced a non-finite value."));
        }

        pixels_[*index] = candidate;
        return {};
    }

    [[nodiscard]] core::Result<Color> resolved_pixel(const std::uint32_t x,
                                                     const std::uint32_t y) const {
        const auto stored = pixel(x, y);
        if (!stored.has_value()) {
            return std::unexpected(stored.error());
        }
        if (stored->weight_sum == Scalar{0}) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "A film pixel with zero total weight cannot be resolved.",
            });
        }

        const auto resolved = Color{
            .red = stored->weighted_sum.red / stored->weight_sum,
            .green = stored->weighted_sum.green / stored->weight_sum,
            .blue = stored->weighted_sum.blue / stored->weight_sum,
        };
        if (!finite(resolved)) {
            return std::unexpected(
                invalid_sample_error("Film resolution produced a non-finite color."));
        }
        return resolved;
    }

  private:
    FilmT(const RenderExtent extent, std::vector<Pixel> pixels) noexcept
        : extent_{extent}, pixels_{std::move(pixels)} {}

    [[nodiscard]] static bool finite(const Color color) noexcept {
        return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue);
    }

    [[nodiscard]] static core::Result<FilmT> allocation_error() {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "The requested film storage could not be allocated.",
        });
    }

    [[nodiscard]] static core::Error invalid_sample_error(const char* const message) {
        return core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = message,
        };
    }

    [[nodiscard]] core::Result<std::size_t> pixel_index(const std::uint32_t x,
                                                        const std::uint32_t y) const {
        if (x >= extent_.width || y >= extent_.height) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Film pixel coordinates are outside the image extent.",
            });
        }
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(extent_.width) +
               static_cast<std::size_t>(x);
    }

    RenderExtent extent_;
    std::vector<Pixel> pixels_;
};

using Film = FilmT<AccumulationPrecision::float32>;
using DoubleAccumulationFilm = FilmT<AccumulationPrecision::float64>;

} // namespace blackframe::renderer
