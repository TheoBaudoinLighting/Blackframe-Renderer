#include <Blackframe/Renderer/FilmTile.hpp>
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace blackframe::renderer {
namespace {

TEST(FilmTileTest, PartitionsTheCropInDeterministicRowMajorOrder) {
    constexpr auto extent = RenderExtent{.width = 8, .height = 6};
    constexpr auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 1, .maximum_x = 6, .maximum_y = 5};
    const auto tiles = make_film_tile_crops(extent, crop, 2);
    ASSERT_TRUE(tiles.has_value());

    const auto expected = std::vector{
        FilmCrop{.minimum_x = 1, .minimum_y = 1, .maximum_x = 3, .maximum_y = 3},
        FilmCrop{.minimum_x = 3, .minimum_y = 1, .maximum_x = 5, .maximum_y = 3},
        FilmCrop{.minimum_x = 5, .minimum_y = 1, .maximum_x = 6, .maximum_y = 3},
        FilmCrop{.minimum_x = 1, .minimum_y = 3, .maximum_x = 3, .maximum_y = 5},
        FilmCrop{.minimum_x = 3, .minimum_y = 3, .maximum_x = 5, .maximum_y = 5},
        FilmCrop{.minimum_x = 5, .minimum_y = 3, .maximum_x = 6, .maximum_y = 5},
    };
    EXPECT_EQ(*tiles, expected);
}

TEST(FilmTileTest, MonolithicAndReversedTileRenderingProduceTheSameCroppedImage) {
    constexpr auto extent = RenderExtent{.width = 8, .height = 6};
    constexpr auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 1, .maximum_x = 6, .maximum_y = 5};
    auto monolithic = Film::create(extent, crop);
    auto tiled = Film::create(extent, crop);
    ASSERT_TRUE(monolithic.has_value());
    ASSERT_TRUE(tiled.has_value());

    const auto tile_crops = make_film_tile_crops(extent, crop, 2);
    ASSERT_TRUE(tile_crops.has_value());
    auto tiles = std::vector<FilmTile>{};
    for (const auto tile_crop : *tile_crops) {
        auto tile = FilmTile::create(extent, tile_crop);
        ASSERT_TRUE(tile.has_value());
        for (std::uint32_t y = tile_crop.minimum_y; y < tile_crop.maximum_y; ++y) {
            for (std::uint32_t x = tile_crop.minimum_x; x < tile_crop.maximum_x; ++x) {
                const auto first = LinearRGB{.red = static_cast<TransportScalar>(x),
                                             .green = static_cast<TransportScalar>(y),
                                             .blue = static_cast<TransportScalar>(x + y)};
                const auto second = LinearRGB{.red = static_cast<TransportScalar>(x + 2),
                                              .green = static_cast<TransportScalar>(y + 2),
                                              .blue = static_cast<TransportScalar>(x + y + 2)};
                ASSERT_TRUE(monolithic->add_sample(x, y, first, 0.5F).has_value());
                ASSERT_TRUE(monolithic->add_sample(x, y, second, 0.5F).has_value());
                ASSERT_TRUE(tile->add_sample(x, y, first, 0.5F).has_value());
                ASSERT_TRUE(tile->add_sample(x, y, second, 0.5F).has_value());
            }
        }
        tiles.push_back(std::move(*tile));
    }

    std::ranges::reverse(tiles);
    ASSERT_TRUE(tiled->merge_tiles(std::span<const FilmTile>{tiles}).has_value());
    EXPECT_EQ(tiled->crop(), crop);
    EXPECT_EQ(tiled->pixel_count(), static_cast<std::size_t>(crop.width() * crop.height()));

    for (std::uint32_t y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (std::uint32_t x = crop.minimum_x; x < crop.maximum_x; ++x) {
            const auto monolithic_pixel = monolithic->pixel(x, y);
            const auto tiled_pixel = tiled->pixel(x, y);
            ASSERT_TRUE(monolithic_pixel.has_value());
            ASSERT_TRUE(tiled_pixel.has_value());
            EXPECT_EQ(*tiled_pixel, *monolithic_pixel);

            const auto monolithic_color = monolithic->resolved_pixel(x, y);
            const auto tiled_color = tiled->resolved_pixel(x, y);
            ASSERT_TRUE(monolithic_color.has_value());
            ASSERT_TRUE(tiled_color.has_value());
            EXPECT_EQ(*tiled_color, *monolithic_color);
        }
    }
    EXPECT_FALSE(tiled->pixel(0, 0).has_value());
    EXPECT_FALSE(tiled->pixel(crop.maximum_x, crop.minimum_y).has_value());
}

TEST(FilmTileTest, RejectsInvalidCropAndIncompleteOrOverlappingFusion) {
    constexpr auto extent = RenderExtent{.width = 4, .height = 2};
    constexpr auto crop = FilmCrop{.maximum_x = 4, .maximum_y = 2};
    EXPECT_FALSE(make_film_tile_crops(extent, crop, 0).has_value());
    EXPECT_FALSE(
        Film::create(extent,
                     FilmCrop{.minimum_x = 2, .minimum_y = 0, .maximum_x = 2, .maximum_y = 2})
            .has_value());

    auto left = FilmTile::create(
        extent, FilmCrop{.minimum_x = 0, .minimum_y = 0, .maximum_x = 3, .maximum_y = 2});
    auto right = FilmTile::create(
        extent, FilmCrop{.minimum_x = 2, .minimum_y = 0, .maximum_x = 4, .maximum_y = 2});
    ASSERT_TRUE(left.has_value());
    ASSERT_TRUE(right.has_value());
    auto overlapping = std::vector<FilmTile>{};
    overlapping.push_back(std::move(*left));
    overlapping.push_back(std::move(*right));

    auto destination = Film::create(extent, crop);
    ASSERT_TRUE(destination.has_value());
    EXPECT_FALSE(destination->merge_tiles(std::span<const FilmTile>{overlapping}).has_value());
    const auto unchanged = destination->pixel(0, 0);
    ASSERT_TRUE(unchanged.has_value());
    EXPECT_EQ(*unchanged, Film::Pixel{});

    auto incomplete_tile = FilmTile::create(
        extent, FilmCrop{.minimum_x = 0, .minimum_y = 0, .maximum_x = 2, .maximum_y = 2});
    ASSERT_TRUE(incomplete_tile.has_value());
    auto incomplete = std::vector<FilmTile>{};
    incomplete.push_back(std::move(*incomplete_tile));
    EXPECT_FALSE(destination->merge_tiles(std::span<const FilmTile>{incomplete}).has_value());
}

TEST(FilmTileTest, RefusesFusionIntoAnAccumulatedDestination) {
    constexpr auto extent = RenderExtent{.width = 2, .height = 1};
    constexpr auto crop = FilmCrop{.maximum_x = 2, .maximum_y = 1};
    auto destination = Film::create(extent, crop);
    auto tile = FilmTile::create(extent, crop);
    ASSERT_TRUE(destination.has_value());
    ASSERT_TRUE(tile.has_value());
    ASSERT_TRUE(destination->add_sample(0, 0, LinearRGB{.red = 1.0F}, 1.0F).has_value());

    const auto before = destination->pixel(0, 0);
    ASSERT_TRUE(before.has_value());
    const auto tiles = std::span<const FilmTile>{&*tile, 1};
    EXPECT_FALSE(destination->merge_tiles(tiles).has_value());
    const auto after = destination->pixel(0, 0);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(*after, *before);
}

} // namespace
} // namespace blackframe::renderer
