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
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::renderer {

struct FilmCrop final {
    std::uint32_t minimum_x{};
    std::uint32_t minimum_y{};
    std::uint32_t maximum_x{};
    std::uint32_t maximum_y{};

    [[nodiscard]] constexpr std::uint32_t width() const noexcept {
        return maximum_x - minimum_x;
    }

    [[nodiscard]] constexpr std::uint32_t height() const noexcept {
        return maximum_y - minimum_y;
    }

    [[nodiscard]] constexpr bool contains(const std::uint32_t x,
                                          const std::uint32_t y) const noexcept {
        return x >= minimum_x && x < maximum_x && y >= minimum_y && y < maximum_y;
    }

    [[nodiscard]] constexpr bool operator==(const FilmCrop&) const noexcept = default;
};

[[nodiscard]] constexpr FilmCrop full_film_crop(const RenderExtent extent) noexcept {
    return FilmCrop{
        .maximum_x = extent.width,
        .maximum_y = extent.height,
    };
}

[[nodiscard]] inline core::Status validate_film_crop(const RenderExtent extent,
                                                     const FilmCrop crop) {
    const auto extent_status = validate_render_extent(extent);
    if (!extent_status.has_value()) {
        return extent_status;
    }
    if (crop.minimum_x >= crop.maximum_x || crop.minimum_y >= crop.maximum_y) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A film crop must have a non-empty half-open pixel range.",
        });
    }
    if (crop.maximum_x > extent.width || crop.maximum_y > extent.height) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A film crop must remain inside the full image extent.",
        });
    }
    return {};
}

template <AccumulationPrecision Precision> struct FilmPixelT final {
    using Scalar = AccumulationScalar<Precision>;
    using Color = LinearRgbT<Scalar>;

    Color weighted_sum{};
    Scalar weight_sum{};
    std::uint64_t sample_count{};

    [[nodiscard]] constexpr bool operator==(const FilmPixelT&) const noexcept = default;
};

namespace film_detail {

template <AccumulationPrecision Precision> struct StoragePixelT;

template <> struct StoragePixelT<AccumulationPrecision::float32> final {
    LinearRgbT<AccumulationScalar<AccumulationPrecision::float32>> weighted_sum{};
    AccumulationScalar<AccumulationPrecision::float32> weight_sum{};
    std::uint64_t sample_count{};
};

template <> struct StoragePixelT<AccumulationPrecision::float64> final {
    LinearRgbT<AccumulationScalar<AccumulationPrecision::float64>> weighted_sum{};
    LinearRgbT<AccumulationScalar<AccumulationPrecision::float64>> weighted_compensation{};
    AccumulationScalar<AccumulationPrecision::float64> weight_sum{};
    AccumulationScalar<AccumulationPrecision::float64> weight_compensation{};
    std::uint64_t sample_count{};
};

} // namespace film_detail

template <AccumulationPrecision Precision> class FilmTileT;

template <AccumulationPrecision Precision> class FilmT final {
  public:
    using Scalar = AccumulationScalar<Precision>;
    using Color = LinearRgbT<Scalar>;
    using Pixel = FilmPixelT<Precision>;

    [[nodiscard]] static core::Result<FilmT> create(const RenderExtent extent) {
        return create(extent, full_film_crop(extent));
    }

    [[nodiscard]] static core::Result<FilmT> create(const RenderExtent extent,
                                                    const FilmCrop crop) {
        const auto crop_status = validate_film_crop(extent, crop);
        if (!crop_status.has_value()) {
            return std::unexpected(crop_status.error());
        }

        const auto pixel_count =
            static_cast<std::size_t>(crop.width()) * static_cast<std::size_t>(crop.height());
        try {
            return FilmT{extent, crop, std::vector<StoragePixel>(pixel_count)};
        } catch (const std::bad_alloc&) {
            return allocation_error();
        } catch (const std::length_error&) {
            return allocation_error();
        }
    }

    [[nodiscard]] constexpr RenderExtent extent() const noexcept {
        return extent_;
    }

    [[nodiscard]] constexpr FilmCrop crop() const noexcept {
        return crop_;
    }

    [[nodiscard]] std::size_t pixel_count() const noexcept {
        return pixels_.size();
    }

    [[nodiscard]] core::Result<Pixel> pixel(const std::uint32_t x, const std::uint32_t y) const {
        const auto index = pixel_index(x, y);
        if (!index.has_value()) {
            return std::unexpected(index.error());
        }
        return snapshot(pixels_[*index]);
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

        auto candidate = current;
        accumulate_sample(candidate, sample, weight);
        ++candidate.sample_count;

        const auto candidate_snapshot = snapshot(candidate);
        if (!finite_storage(candidate) || !finite(candidate_snapshot.weighted_sum) ||
            !std::isfinite(candidate_snapshot.weight_sum)) {
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

    [[nodiscard]] core::Status merge_tiles(std::span<const FilmTileT<Precision>> tiles);

  private:
    using StoragePixel = film_detail::StoragePixelT<Precision>;

    FilmT(const RenderExtent extent, const FilmCrop crop, std::vector<StoragePixel> pixels) noexcept
        : extent_{extent}, crop_{crop}, pixels_{std::move(pixels)} {}

    [[nodiscard]] static bool finite(const Color color) noexcept {
        return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue);
    }

    // Neumaier compensation recovers terms smaller than the current double-precision sum.
    static void accumulate(Scalar& sum, Scalar& compensation, const Scalar value) noexcept {
        const auto updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            compensation += (sum - updated) + value;
        } else {
            compensation += (value - updated) + sum;
        }
        sum = updated;
    }

    static void accumulate_sample(StoragePixel& stored, const Color sample,
                                  const Scalar weight) noexcept {
        if constexpr (Precision == AccumulationPrecision::float64) {
            accumulate(stored.weighted_sum.red, stored.weighted_compensation.red,
                       sample.red * weight);
            accumulate(stored.weighted_sum.green, stored.weighted_compensation.green,
                       sample.green * weight);
            accumulate(stored.weighted_sum.blue, stored.weighted_compensation.blue,
                       sample.blue * weight);
            accumulate(stored.weight_sum, stored.weight_compensation, weight);
        } else {
            stored.weighted_sum.red += sample.red * weight;
            stored.weighted_sum.green += sample.green * weight;
            stored.weighted_sum.blue += sample.blue * weight;
            stored.weight_sum += weight;
        }
    }

    [[nodiscard]] static Pixel snapshot(const StoragePixel& stored) noexcept {
        if constexpr (Precision == AccumulationPrecision::float64) {
            return Pixel{
                .weighted_sum =
                    Color{
                        .red = stored.weighted_sum.red + stored.weighted_compensation.red,
                        .green = stored.weighted_sum.green + stored.weighted_compensation.green,
                        .blue = stored.weighted_sum.blue + stored.weighted_compensation.blue,
                    },
                .weight_sum = stored.weight_sum + stored.weight_compensation,
                .sample_count = stored.sample_count,
            };
        } else {
            return Pixel{
                .weighted_sum = stored.weighted_sum,
                .weight_sum = stored.weight_sum,
                .sample_count = stored.sample_count,
            };
        }
    }

    [[nodiscard]] static bool finite_storage(const StoragePixel& stored) noexcept {
        if constexpr (Precision == AccumulationPrecision::float64) {
            return finite(stored.weighted_sum) && finite(stored.weighted_compensation) &&
                   std::isfinite(stored.weight_sum) && std::isfinite(stored.weight_compensation);
        } else {
            return finite(stored.weighted_sum) && std::isfinite(stored.weight_sum);
        }
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
        if (!crop_.contains(x, y)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Film pixel coordinates are outside the active crop.",
            });
        }
        return static_cast<std::size_t>(y - crop_.minimum_y) *
                   static_cast<std::size_t>(crop_.width()) +
               static_cast<std::size_t>(x - crop_.minimum_x);
    }

    friend class FilmTileT<Precision>;

    RenderExtent extent_;
    FilmCrop crop_;
    std::vector<StoragePixel> pixels_;
};

using Film = FilmT<AccumulationPrecision::float32>;
using ReferenceFilm = FilmT<AccumulationPrecision::float64>;
using DoubleAccumulationFilm = ReferenceFilm;

} // namespace blackframe::renderer
