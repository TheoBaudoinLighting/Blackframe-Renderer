#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <Blackframe/Renderer/HostUdimTexture.hpp>
#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

struct Rgb8 final {
    std::uint8_t red{};
    std::uint8_t green{};
    std::uint8_t blue{};
};

inline constexpr auto GridColumns = std::uint32_t{4U};
inline constexpr auto GridRows = std::uint32_t{2U};
inline constexpr auto TileExtent = std::uint32_t{32U};
inline constexpr auto GridPalette = std::array{
    Rgb8{.red = 224U, .green = 52U, .blue = 48U},  Rgb8{.red = 54U, .green = 190U, .blue = 75U},
    Rgb8{.red = 55U, .green = 92U, .blue = 220U},  Rgb8{.red = 225U, .green = 181U, .blue = 45U},
    Rgb8{.red = 205U, .green = 59U, .blue = 188U}, Rgb8{.red = 43U, .green = 190U, .blue = 198U},
    Rgb8{.red = 225U, .green = 113U, .blue = 38U}, Rgb8{.red = 114U, .green = 72U, .blue = 205U},
};

[[nodiscard]] std::filesystem::path artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::filesystem::path udim_pattern(const std::string_view prefix) {
    return artifact_path(std::string{prefix} + ".<UDIM>.ppm");
}

[[nodiscard]] std::filesystem::path tile_path(const std::string_view prefix,
                                              const std::uint32_t number) {
    return artifact_path(std::string{prefix} + "." + std::to_string(number) + ".ppm");
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

[[nodiscard]] ReferenceScalar decode_srgb_byte(const std::uint8_t byte) {
    const auto encoded = static_cast<ReferenceScalar>(byte) / ReferenceScalar{255};
    return encoded <= ReferenceScalar{0.04045}
               ? encoded / ReferenceScalar{12.92}
               : std::pow((encoded + ReferenceScalar{0.055}) / ReferenceScalar{1.055},
                          ReferenceScalar{2.4});
}

[[nodiscard]] Rgb8 tile_pixel(const Rgb8 base, const std::uint32_t x,
                              const std::uint32_t y) noexcept {
    constexpr auto border = std::uint32_t{2U};
    if (x < border || y < border || x >= TileExtent - border || y >= TileExtent - border) {
        return Rgb8{.red = 7U, .green = 7U, .blue = 9U};
    }
    if (x < 7U && y < 7U) {
        return Rgb8{.red = 255U, .green = 255U, .blue = 255U};
    }
    if (x >= TileExtent - 7U && y < 7U) {
        return Rgb8{.red = 255U, .green = 30U, .blue = 30U};
    }
    if (x < 7U && y >= TileExtent - 7U) {
        return Rgb8{.red = 30U, .green = 255U, .blue = 30U};
    }
    if (x >= TileExtent - 7U && y >= TileExtent - 7U) {
        return Rgb8{.red = 30U, .green = 30U, .blue = 255U};
    }
    if (x == y || x + 1U == y) {
        return Rgb8{
            .red = static_cast<std::uint8_t>((static_cast<std::uint32_t>(base.red) + 255U) / 2U),
            .green =
                static_cast<std::uint8_t>((static_cast<std::uint32_t>(base.green) + 255U) / 2U),
            .blue = static_cast<std::uint8_t>((static_cast<std::uint32_t>(base.blue) + 255U) / 2U),
        };
    }
    return base;
}

[[nodiscard]] std::filesystem::path write_tile(const std::string_view prefix,
                                               const std::uint32_t number, const Rgb8 base) {
    const auto output = tile_path(prefix, number);
    auto pixels = std::vector<std::uint8_t>{};
    pixels.reserve(static_cast<std::size_t>(TileExtent) * TileExtent * 3U);
    for (auto y = std::uint32_t{}; y < TileExtent; ++y) {
        for (auto x = std::uint32_t{}; x < TileExtent; ++x) {
            const auto pixel = tile_pixel(base, x, y);
            pixels.push_back(pixel.red);
            pixels.push_back(pixel.green);
            pixels.push_back(pixel.blue);
        }
    }

    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "P6\n" << TileExtent << ' ' << TileExtent << "\n255\n";
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));
    EXPECT_TRUE(stream.good());
    return output;
}

void write_grid_tiles(const std::string_view prefix) {
    for (auto row = std::uint32_t{}; row < GridRows; ++row) {
        for (auto column = std::uint32_t{}; column < GridColumns; ++column) {
            const auto index = static_cast<std::size_t>(row * GridColumns + column);
            const auto number = UdimFirstTileNumber + row * UdimColumnsPerRow + column;
            const auto output = write_tile(prefix, number, GridPalette[index]);
            EXPECT_TRUE(std::filesystem::is_regular_file(output));
        }
    }
}

void expect_address(const UdimAddress& address, const std::uint32_t number,
                    const std::uint32_t column, const std::uint32_t row, const Point2 local_uv) {
    EXPECT_EQ(address.tile, (UdimTile{.number = number, .column = column, .row = row}));
    EXPECT_EQ(address.local_uv, local_uv);
}

void expect_address(const ReferenceUdimAddress& address, const std::uint32_t number,
                    const std::uint32_t column, const std::uint32_t row,
                    const ReferencePoint2 local_uv) {
    EXPECT_EQ(address.tile, (UdimTile{.number = number, .column = column, .row = row}));
    EXPECT_EQ(address.local_uv, local_uv);
}

TEST(HostUdimTextureTest, KeepsTheAddressContractStable) {
    static_assert(std::is_standard_layout_v<UdimTile>);
    static_assert(std::is_trivially_copyable_v<UdimTile>);
    static_assert(std::is_standard_layout_v<UdimAddress>);
    static_assert(std::is_trivially_copyable_v<UdimAddress>);
    static_assert(std::is_standard_layout_v<ReferenceUdimAddress>);
    static_assert(std::is_trivially_copyable_v<ReferenceUdimAddress>);
    EXPECT_EQ(sizeof(UdimTile), 3U * sizeof(std::uint32_t));
    EXPECT_EQ(UdimFirstTileNumber, 1001U);
    EXPECT_EQ(UdimColumnsPerRow, 10U);
    EXPECT_EQ(UdimMaximumColumn, 9U);
}

TEST(HostUdimTextureTest, ResolvesStandardBoundariesAndMatchesReferencePrecision) {
    const auto origin = resolve_udim_address(Point2{});
    ASSERT_TRUE(origin.has_value()) << origin.error().message;
    expect_address(*origin, 1001U, 0U, 0U, Point2{});

    const auto below_one = std::nextafter(TransportScalar{1}, TransportScalar{});
    const auto below_boundary = resolve_udim_address(Point2{.x = below_one, .y = below_one});
    ASSERT_TRUE(below_boundary.has_value()) << below_boundary.error().message;
    expect_address(*below_boundary, 1001U, 0U, 0U, Point2{.x = below_one, .y = below_one});

    const auto u_boundary = resolve_udim_address(Point2{.x = 1.0F, .y = 0.25F});
    const auto v_boundary = resolve_udim_address(Point2{.x = 0.25F, .y = 1.0F});
    const auto last_column = resolve_udim_address(Point2{.x = 9.25F, .y = 3.75F});
    ASSERT_TRUE(u_boundary.has_value()) << u_boundary.error().message;
    ASSERT_TRUE(v_boundary.has_value()) << v_boundary.error().message;
    ASSERT_TRUE(last_column.has_value()) << last_column.error().message;
    expect_address(*u_boundary, 1002U, 1U, 0U, Point2{.x = 0.0F, .y = 0.25F});
    expect_address(*v_boundary, 1011U, 0U, 1U, Point2{.x = 0.25F, .y = 0.0F});
    expect_address(*last_column, 1040U, 9U, 3U, Point2{.x = 0.25F, .y = 0.75F});

    const auto transport = resolve_udim_address(Point2{.x = 3.25F, .y = 2.75F});
    const auto reference = resolve_udim_address(ReferencePoint2{.x = 3.25, .y = 2.75});
    ASSERT_TRUE(transport.has_value()) << transport.error().message;
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    expect_address(*transport, 1024U, 3U, 2U, Point2{.x = 0.25F, .y = 0.75F});
    expect_address(*reference, 1024U, 3U, 2U, ReferencePoint2{.x = 0.25, .y = 0.75});

    constexpr auto maximum_number = std::numeric_limits<std::uint32_t>::max();
    constexpr auto maximum_number_column =
        (maximum_number - UdimFirstTileNumber) % UdimColumnsPerRow;
    constexpr auto maximum_row =
        (maximum_number - UdimFirstTileNumber - maximum_number_column) / UdimColumnsPerRow;
    const auto maximum = resolve_udim_address(
        ReferencePoint2{.x = static_cast<ReferenceScalar>(maximum_number_column) + 0.25,
                        .y = static_cast<ReferenceScalar>(maximum_row) + 0.5});
    ASSERT_TRUE(maximum.has_value()) << maximum.error().message;
    expect_address(*maximum, maximum_number, maximum_number_column, maximum_row,
                   ReferencePoint2{.x = 0.25, .y = 0.5});
    const auto row_overflow = resolve_udim_address(
        ReferencePoint2{.x = static_cast<ReferenceScalar>(maximum_number_column),
                        .y = static_cast<ReferenceScalar>(maximum_row) + 1.0});
    ASSERT_FALSE(row_overflow.has_value());
    EXPECT_EQ(row_overflow.error().code, core::StatusCode::invalid_argument);
}

TEST(HostUdimTextureTest, RejectsInvalidCoordinatesWithoutAliasingOrClamping) {
    constexpr auto invalid_transport = std::array{
        Point2{.x = -0.01F, .y = 0.0F},
        Point2{.x = 0.0F, .y = -0.01F},
        Point2{.x = 10.0F, .y = 0.0F},
        Point2{.x = std::numeric_limits<TransportScalar>::quiet_NaN(), .y = 0.0F},
        Point2{.x = 0.0F, .y = std::numeric_limits<TransportScalar>::infinity()},
        Point2{.x = 0.0F, .y = std::numeric_limits<TransportScalar>::max()},
    };
    for (const auto uv : invalid_transport) {
        const auto resolved = resolve_udim_address(uv);
        ASSERT_FALSE(resolved.has_value());
        EXPECT_EQ(resolved.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(resolved.error().message.empty());
    }

    constexpr auto invalid_reference = std::array{
        ReferencePoint2{.x = -0.01, .y = 0.0},
        ReferencePoint2{.x = 10.0, .y = 0.0},
        ReferencePoint2{.x = 0.0, .y = std::numeric_limits<ReferenceScalar>::quiet_NaN()},
        ReferencePoint2{.x = 0.0, .y = std::numeric_limits<ReferenceScalar>::infinity()},
        ReferencePoint2{.x = 0.0, .y = std::numeric_limits<ReferenceScalar>::max()},
    };
    for (const auto uv : invalid_reference) {
        const auto resolved = resolve_udim_address(uv);
        ASSERT_FALSE(resolved.has_value());
        EXPECT_EQ(resolved.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(resolved.error().message.empty());
    }
}

TEST(HostUdimTextureTest, RequiresOneAbsoluteTokenAndAnExplicitColorSpace) {
    const auto valid_pattern = udim_pattern("texture-udim-contract");
    const auto created = HostUdimTexture::create(valid_pattern, TextureColorSpace::data);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    EXPECT_EQ(created->pattern(), valid_pattern.lexically_normal());
    EXPECT_EQ(created->source_color_space(), TextureColorSpace::data);

    const auto relative = HostUdimTexture::create(std::filesystem::path{"relative.<UDIM>.ppm"},
                                                  TextureColorSpace::data);
    const auto missing_token = HostUdimTexture::create(
        artifact_path("texture-udim-contract.1001.ppm"), TextureColorSpace::data);
    const auto duplicate_token = HostUdimTexture::create(
        artifact_path("texture-udim-<UDIM>-<UDIM>.ppm"), TextureColorSpace::data);
    const auto invalid_color = HostUdimTexture::create(
        valid_pattern, static_cast<TextureColorSpace>(std::numeric_limits<std::uint32_t>::max()));
    for (const auto* result : {&relative, &missing_token, &duplicate_token, &invalid_color}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_EQ(result->error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result->error().message.empty());
    }
}

TEST(HostUdimTextureTest, ResolvesACompleteFourByTwoGridAndReusesCacheHandles) {
    constexpr auto prefix = std::string_view{"texture-udim-grid-unit"};
    write_grid_tiles(prefix);
    const auto texture = HostUdimTexture::create(udim_pattern(prefix), TextureColorSpace::data);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;

    auto first_handle = HostImageHandle{};
    for (auto row = std::uint32_t{}; row < GridRows; ++row) {
        for (auto column = std::uint32_t{}; column < GridColumns; ++column) {
            const auto number = UdimFirstTileNumber + row * UdimColumnsPerRow + column;
            const auto uv = Point2{.x = static_cast<TransportScalar>(column) + 0.5F,
                                   .y = static_cast<TransportScalar>(row) + 0.5F};
            const auto resolved = texture->resolve(*cache, uv);
            ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
            expect_address(resolved->address, number, column, row, Point2{.x = 0.5F, .y = 0.5F});
            EXPECT_EQ(resolved->image->source_path(),
                      std::filesystem::canonical(tile_path(prefix, number)));
            EXPECT_EQ(resolved->image->width(), TileExtent);
            EXPECT_EQ(resolved->image->height(), TileExtent);
            EXPECT_EQ(resolved->image->channel_count(), 3U);
            if (number == UdimFirstTileNumber) {
                first_handle = resolved->image;
            }
        }
    }

    const auto reference = texture->resolve(*cache, ReferencePoint2{.x = 0.5, .y = 0.5});
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    expect_address(reference->address, 1001U, 0U, 0U, ReferencePoint2{.x = 0.5, .y = 0.5});
    EXPECT_EQ(reference->image.get(), first_handle.get());
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value()) << entry_count.error().message;
    EXPECT_EQ(*entry_count, static_cast<std::size_t>(GridColumns * GridRows));
}

TEST(HostUdimTextureTest, PreservesTheExplicitSrgbTagThroughTileLoading) {
    constexpr auto prefix = std::string_view{"texture-udim-srgb"};
    ASSERT_TRUE(std::filesystem::is_regular_file(write_tile(prefix, 1001U, GridPalette[0])));
    const auto texture = HostUdimTexture::create(udim_pattern(prefix), TextureColorSpace::srgb);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;

    const auto resolved = texture->resolve(*cache, Point2{.x = 0.25F, .y = 0.25F});
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->image->source_color_space(), TextureColorSpace::srgb);
    EXPECT_EQ(resolved->image->storage_color_space(), TextureColorSpace::scene_linear_srgb);
    constexpr auto sample_x = std::uint32_t{8U};
    constexpr auto sample_y = std::uint32_t{10U};
    const auto pixel_offset = static_cast<std::size_t>((sample_y * TileExtent + sample_x) * 3U);
    ASSERT_GE(resolved->image->pixels().size(), pixel_offset + 3U);
    EXPECT_NEAR(resolved->image->pixels()[pixel_offset], decode_srgb_byte(GridPalette[0].red),
                2.0e-7);
    EXPECT_NEAR(resolved->image->pixels()[pixel_offset + 1U],
                decode_srgb_byte(GridPalette[0].green), 2.0e-7);
    EXPECT_NEAR(resolved->image->pixels()[pixel_offset + 2U], decode_srgb_byte(GridPalette[0].blue),
                2.0e-7);
}

TEST(HostUdimTextureTest, PreservesUnicodePathsInLoadsAndMissingTileDiagnostics) {
    const auto unicode_directory =
        artifact_path("texture-udim-unicode") / std::filesystem::path{u8"r\u00E9pertoire"};
    auto filesystem_error = std::error_code{};
    std::filesystem::create_directories(unicode_directory, filesystem_error);
    ASSERT_FALSE(filesystem_error) << filesystem_error.message();

    const auto unicode_pattern =
        unicode_directory / std::filesystem::path{u8"grille-\u00E9.<UDIM>.ppm"};
    const auto first_path = unicode_directory / std::filesystem::path{u8"grille-\u00E9.1001.ppm"};
    const auto missing_path = unicode_directory / std::filesystem::path{u8"grille-\u00E9.1002.ppm"};
    const auto source_path = write_tile("texture-udim-unicode-source", 1001U, GridPalette[0]);
    std::filesystem::copy_file(source_path, first_path,
                               std::filesystem::copy_options::overwrite_existing, filesystem_error);
    ASSERT_FALSE(filesystem_error) << filesystem_error.message();
    filesystem_error.clear();
    std::filesystem::remove(missing_path, filesystem_error);
    ASSERT_FALSE(filesystem_error) << filesystem_error.message();

    const auto texture = HostUdimTexture::create(unicode_pattern, TextureColorSpace::data);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto first = texture->resolve(*cache, Point2{.x = 0.5F, .y = 0.5F});
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_EQ(first->image->source_path(), std::filesystem::canonical(first_path));

    const auto missing = texture->resolve(*cache, Point2{.x = 1.5F, .y = 0.5F});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);
    EXPECT_NE(missing.error().message.find("1002"), std::string::npos);
    auto preferred_missing_path = missing_path.lexically_normal();
    preferred_missing_path.make_preferred();
    EXPECT_NE(missing.error().message.find(utf8_path(preferred_missing_path)), std::string::npos)
        << missing.error().message << '\n'
        << utf8_path(preferred_missing_path);
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);
}

TEST(HostUdimTextureTest, ReportsMissingTileNumberAndPathWithoutNeighborFallback) {
    constexpr auto prefix = std::string_view{"texture-udim-missing"};
    const auto first_path = write_tile(prefix, 1001U, GridPalette[0]);
    const auto third_path = write_tile(prefix, 1003U, GridPalette[2]);
    ASSERT_TRUE(std::filesystem::is_regular_file(first_path));
    ASSERT_TRUE(std::filesystem::is_regular_file(third_path));
    const auto missing_path = tile_path(prefix, 1002U).lexically_normal();
    auto remove_error = std::error_code{};
    std::filesystem::remove(missing_path, remove_error);
    ASSERT_FALSE(remove_error);

    const auto texture = HostUdimTexture::create(udim_pattern(prefix), TextureColorSpace::data);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto first = texture->resolve(*cache, Point2{.x = 0.5F, .y = 0.5F});
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto before = cache->entry_count();
    ASSERT_TRUE(before.has_value());
    ASSERT_EQ(*before, 1U);

    const auto missing = texture->resolve(*cache, Point2{.x = 1.5F, .y = 0.5F});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);
    EXPECT_NE(missing.error().message.find("1002"), std::string::npos);
    EXPECT_NE(missing.error().message.find(missing_path.string()), std::string::npos);
    const auto after = cache->entry_count();
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(*after, *before);
    EXPECT_EQ(first->image->source_path(), std::filesystem::canonical(first_path));

    const auto third = texture->resolve(*cache, Point2{.x = 2.5F, .y = 0.5F});
    ASSERT_TRUE(third.has_value()) << third.error().message;
    EXPECT_EQ(third->address.tile.number, 1003U);
    EXPECT_EQ(third->image->source_path(), std::filesystem::canonical(third_path));
}

TEST(HostUdimTextureTest, PropagatesCacheBudgetsInsteadOfReportingAFalseAbsence) {
    constexpr auto prefix = std::string_view{"texture-udim-budget"};
    ASSERT_TRUE(std::filesystem::is_regular_file(write_tile(prefix, 1001U, GridPalette[0])));
    ASSERT_TRUE(std::filesystem::is_regular_file(write_tile(prefix, 1002U, GridPalette[1])));
    const auto texture = HostUdimTexture::create(udim_pattern(prefix), TextureColorSpace::data);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto limits = HostImageCacheLimits{};
    limits.maximum_entries = 1U;
    auto cache = HostImageCache::create(limits);
    ASSERT_TRUE(cache.has_value()) << cache.error().message;

    const auto first = texture->resolve(*cache, Point2{.x = 0.5F, .y = 0.5F});
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto exhausted = texture->resolve(*cache, Point2{.x = 1.5F, .y = 0.5F});
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error().code, core::StatusCode::resource_exhausted);
    EXPECT_NE(exhausted.error().message.find("1002"), std::string::npos);
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);
}

[[nodiscard]] std::filesystem::path checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return {};
    }
    const auto output = value_size > 1U ? std::filesystem::path{value} : std::filesystem::path{};
    std::free(value);
    return output;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    return value == nullptr || *value == '\0' ? std::filesystem::path{}
                                              : std::filesystem::path{value};
#endif
}

#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
[[nodiscard]] core::Status write_udim_grid_preview(const HostUdimTexture& texture,
                                                   HostImageCache& cache,
                                                   const std::filesystem::path& output) {
    constexpr auto output_extent = std::uint32_t{800U};
    constexpr auto cell_extent = output_extent / GridColumns;
    constexpr auto grid_height = cell_extent * GridRows;
    constexpr auto grid_top = (output_extent - grid_height) / 2U;
    auto resolved_tiles = std::array<HostImageHandle, GridColumns * GridRows>{};
    for (auto row = std::uint32_t{}; row < GridRows; ++row) {
        for (auto column = std::uint32_t{}; column < GridColumns; ++column) {
            const auto resolved =
                texture.resolve(cache, Point2{.x = static_cast<TransportScalar>(column) + 0.5F,
                                              .y = static_cast<TransportScalar>(row) + 0.5F});
            if (!resolved) {
                return std::unexpected(resolved.error());
            }
            resolved_tiles[static_cast<std::size_t>(row * GridColumns + column)] = resolved->image;
        }
    }

    auto film = Film::create(RenderExtent{.width = output_extent, .height = output_extent});
    if (!film) {
        return std::unexpected(film.error());
    }
    for (auto y = std::uint32_t{}; y < output_extent; ++y) {
        for (auto x = std::uint32_t{}; x < output_extent; ++x) {
            auto color = LinearRGB{.red = 0.012F, .green = 0.014F, .blue = 0.020F};
            if (y >= grid_top && y < grid_top + grid_height) {
                const auto column = x / cell_extent;
                const auto row = (y - grid_top) / cell_extent;
                const auto local_uv = Point2{
                    .x = (static_cast<TransportScalar>(x % cell_extent) + 0.5F) /
                         static_cast<TransportScalar>(cell_extent),
                    .y = (static_cast<TransportScalar>((y - grid_top) % cell_extent) + 0.5F) /
                         static_cast<TransportScalar>(cell_extent),
                };
                const auto& image =
                    resolved_tiles[static_cast<std::size_t>(row * GridColumns + column)];
                auto channels = std::array<core::Result<TransportScalar>, 3>{
                    filter_host_image_channel(*image, local_uv, 0U, TextureFilterMode::nearest,
                                              TextureWrapMode::clamp, TextureWrapMode::clamp),
                    filter_host_image_channel(*image, local_uv, 1U, TextureFilterMode::nearest,
                                              TextureWrapMode::clamp, TextureWrapMode::clamp),
                    filter_host_image_channel(*image, local_uv, 2U, TextureFilterMode::nearest,
                                              TextureWrapMode::clamp, TextureWrapMode::clamp),
                };
                for (const auto& channel : channels) {
                    if (!channel) {
                        return std::unexpected(channel.error());
                    }
                }
                color = LinearRGB{.red = *channels[0], .green = *channels[1], .blue = *channels[2]};
            }
            const auto accumulated = film->add_sample(x, y, color, TransportScalar{1});
            if (!accumulated) {
                return accumulated;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(HostUdimTextureTest, WritesStableMultiUdimGrid) {
    const auto output = checksum_output_path();
    if (output.empty()) {
        GTEST_SKIP() << "The explicit PNG checksum output path was not supplied.";
    }
#if !defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
    FAIL() << "The UDIM grid preview requires the explicit PNG capability.";
#else
    constexpr auto prefix = std::string_view{"texture-udim-grid-preview"};
    write_grid_tiles(prefix);
    const auto texture = HostUdimTexture::create(udim_pattern(prefix), TextureColorSpace::data);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto written = write_udim_grid_preview(*texture, *cache, output);
    ASSERT_TRUE(written.has_value()) << written.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output));
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, static_cast<std::size_t>(GridColumns * GridRows));
    testing::Test::RecordProperty(
        "grid_layout",
        "top=1001,1002,1003,1004;bottom=1011,1012,1013,1014;white=top-left;red=top-right;"
        "green=bottom-left;blue=bottom-right");
#endif
}

} // namespace
} // namespace blackframe::renderer
