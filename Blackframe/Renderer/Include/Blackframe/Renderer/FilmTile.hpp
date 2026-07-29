#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::renderer {

[[nodiscard]] inline core::Result<std::vector<FilmCrop>>
make_film_tile_crops(const RenderExtent extent, const FilmCrop crop,
                     const std::uint32_t tile_edge_length) {
    const auto crop_status = validate_film_crop(extent, crop);
    if (!crop_status.has_value()) {
        return std::unexpected(crop_status.error());
    }
    if (tile_edge_length == 0) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Film tile edge length must be greater than zero.",
        });
    }

    try {
        auto tiles = std::vector<FilmCrop>{};
        const auto horizontal_count =
            (static_cast<std::uint64_t>(crop.width()) + tile_edge_length - 1U) / tile_edge_length;
        const auto vertical_count =
            (static_cast<std::uint64_t>(crop.height()) + tile_edge_length - 1U) / tile_edge_length;
        tiles.reserve(static_cast<std::size_t>(horizontal_count * vertical_count));

        for (auto minimum_y = crop.minimum_y; minimum_y < crop.maximum_y;) {
            const auto maximum_y = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                crop.maximum_y, static_cast<std::uint64_t>(minimum_y) + tile_edge_length));
            for (auto minimum_x = crop.minimum_x; minimum_x < crop.maximum_x;) {
                const auto maximum_x = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    crop.maximum_x, static_cast<std::uint64_t>(minimum_x) + tile_edge_length));
                tiles.push_back(FilmCrop{
                    .minimum_x = minimum_x,
                    .minimum_y = minimum_y,
                    .maximum_x = maximum_x,
                    .maximum_y = maximum_y,
                });
                minimum_x = maximum_x;
            }
            minimum_y = maximum_y;
        }
        return tiles;
    } catch (const std::bad_alloc&) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Film tile metadata could not be allocated.",
        });
    } catch (const std::length_error&) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Film tile metadata exceeds the supported container size.",
        });
    }
}

template <AccumulationPrecision Precision> class FilmTileT final {
  public:
    using Scalar = typename FilmT<Precision>::Scalar;
    using Color = typename FilmT<Precision>::Color;
    using Pixel = typename FilmT<Precision>::Pixel;

    [[nodiscard]] static core::Result<FilmTileT> create(const RenderExtent extent,
                                                        const FilmCrop crop) {
        auto storage = FilmT<Precision>::create(extent, crop);
        if (!storage.has_value()) {
            return std::unexpected(storage.error());
        }
        return FilmTileT{std::move(*storage)};
    }

    [[nodiscard]] constexpr RenderExtent extent() const noexcept {
        return storage_.extent();
    }

    [[nodiscard]] constexpr FilmCrop crop() const noexcept {
        return storage_.crop();
    }

    [[nodiscard]] std::size_t pixel_count() const noexcept {
        return storage_.pixel_count();
    }

    [[nodiscard]] core::Status add_sample(const std::uint32_t x, const std::uint32_t y,
                                          const Color sample, const Scalar weight) {
        return storage_.add_sample(x, y, sample, weight);
    }

    [[nodiscard]] core::Result<Pixel> pixel(const std::uint32_t x, const std::uint32_t y) const {
        return storage_.pixel(x, y);
    }

  private:
    explicit FilmTileT(FilmT<Precision> storage) noexcept : storage_{std::move(storage)} {}

    friend class FilmT<Precision>;

    FilmT<Precision> storage_;
};

template <AccumulationPrecision Precision>
core::Status FilmT<Precision>::merge_tiles(const std::span<const FilmTileT<Precision>> tiles) {
    if (tiles.empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Film tile fusion requires a complete non-empty tile partition.",
        });
    }
    if (std::ranges::any_of(pixels_,
                            [](const StoragePixel& pixel) { return pixel.sample_count != 0; })) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Film tiles can only be fused into a pristine destination crop.",
        });
    }

    try {
        auto canonical_order = std::vector<std::size_t>(tiles.size());
        std::iota(canonical_order.begin(), canonical_order.end(), std::size_t{0});
        std::ranges::stable_sort(canonical_order, {}, [&](const std::size_t index) {
            const auto tile_crop = tiles[index].crop();
            return std::array{
                tile_crop.minimum_y,
                tile_crop.minimum_x,
                tile_crop.maximum_y,
                tile_crop.maximum_x,
            };
        });

        auto candidate = std::vector<StoragePixel>(pixels_.size());
        auto covered = std::vector<std::uint8_t>(pixels_.size(), std::uint8_t{0});
        for (const auto tile_index : canonical_order) {
            const auto& tile = tiles[tile_index].storage_;
            const auto tile_extent = tile.extent();
            const auto tile_crop = tile.crop();
            if (tile_extent.width != extent_.width || tile_extent.height != extent_.height ||
                tile_crop.minimum_x < crop_.minimum_x || tile_crop.minimum_y < crop_.minimum_y ||
                tile_crop.maximum_x > crop_.maximum_x || tile_crop.maximum_y > crop_.maximum_y) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "Every film tile must belong to the destination extent and crop.",
                });
            }

            for (std::uint32_t y = tile_crop.minimum_y; y < tile_crop.maximum_y; ++y) {
                for (std::uint32_t x = tile_crop.minimum_x; x < tile_crop.maximum_x; ++x) {
                    const auto destination_index =
                        static_cast<std::size_t>(y - crop_.minimum_y) * crop_.width() +
                        static_cast<std::size_t>(x - crop_.minimum_x);
                    if (covered[destination_index] != 0) {
                        return std::unexpected(core::Error{
                            .code = core::StatusCode::invalid_argument,
                            .message = "Film tile crops must not overlap.",
                        });
                    }

                    const auto source_index =
                        static_cast<std::size_t>(y - tile_crop.minimum_y) * tile_crop.width() +
                        static_cast<std::size_t>(x - tile_crop.minimum_x);
                    candidate[destination_index] = tile.pixels_[source_index];
                    covered[destination_index] = 1;
                }
            }
        }

        if (std::ranges::any_of(covered, [](const std::uint8_t value) { return value == 0; })) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Film tile crops must cover the destination crop exactly.",
            });
        }

        pixels_ = std::move(candidate);
        return {};
    } catch (const std::bad_alloc&) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Deterministic film tile fusion could not allocate its staging storage.",
        });
    } catch (const std::length_error&) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "Deterministic film tile fusion exceeds the supported container size.",
        });
    }
}

using FilmTile = FilmTileT<AccumulationPrecision::float32>;
using ReferenceFilmTile = FilmTileT<AccumulationPrecision::float64>;

} // namespace blackframe::renderer
