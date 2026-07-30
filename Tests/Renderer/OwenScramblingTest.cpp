#include <Blackframe/Renderer/OwenScrambling.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <system_error>
#include <type_traits>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

struct KnownOwenBits final {
    SampleStreamIndex index;
    std::uint64_t dimension;
    std::uint64_t bits;
};

constexpr auto known_owen_bits = std::array{
    KnownOwenBits{
        .index = {},
        .dimension = 0,
        .bits = 0x92A8B3FFD11450D8ULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 1},
        .dimension = 0,
        .bits = 0x05425D9FC8F958EFULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 2, .seed = 0x0123456789ABCDEFULL},
        .dimension = 0,
        .bits = 0x2D5F4AEE875D131DULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 0x0123456789ABCDEFULL, .seed = 0x0123456789ABCDEFULL},
        .dimension = 63,
        .bits = 0x952C2015E62C23E3ULL,
    },
    KnownOwenBits{
        .index = {.pixel_x = 113,
                  .pixel_y = 47,
                  .sample_index = 0x0123456789ABCDEFULL,
                  .seed = 0xA5A5F00D12345678ULL},
        .dimension = 63,
        .bits = 0xA694114D8B9547EFULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 0x0123456789ABCDEFULL, .seed = 0xFEDCBA9876543210ULL},
        .dimension = 255,
        .bits = 0x210EA3A92DEE7EFCULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 0x0123456789ABCDEFULL},
        .dimension = 21'200,
        .bits = 0x8A185723D91F9C0BULL,
    },
    KnownOwenBits{
        .index = {.sample_index = 0x8000000000000000ULL,
                  .seed = std::numeric_limits<std::uint64_t>::max()},
        .dimension = 21'200,
        .bits = 0x5E46341BD2FA3170ULL,
    },
    KnownOwenBits{
        .index = {.sample_index = std::numeric_limits<std::uint64_t>::max(),
                  .seed = 0x0123456789ABCDEFULL},
        .dimension = 21'200,
        .bits = 0x3731AD8327D19D7CULL,
    },
};

[[nodiscard]] constexpr std::size_t leading_bin(const std::uint64_t bits,
                                                const std::uint32_t bin_bits) noexcept {
    if (bin_bits == 0) {
        return 0;
    }
    return static_cast<std::size_t>(bits >> (64U - bin_bits));
}

void expect_exact_owen_projection(const SampleStreamIndex base_index,
                                  const std::uint64_t x_dimension, const std::uint64_t y_dimension,
                                  const std::span<const std::uint64_t> block_offsets,
                                  const std::uint32_t partition_bits,
                                  const std::uint16_t expected_per_cell) {
    constexpr auto sample_count = std::size_t{4'096};
    const auto cell_count = std::size_t{1} << partition_bits;

    for (const auto block_offset : block_offsets) {
        std::vector<std::uint64_t> x_samples(sample_count);
        std::vector<std::uint64_t> y_samples(sample_count);
        for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
            auto index = base_index;
            index.sample_index = block_offset + static_cast<std::uint64_t>(sample);
            const auto x = owen_scrambled_sobol_bits(index, x_dimension);
            const auto y = owen_scrambled_sobol_bits(index, y_dimension);
            ASSERT_TRUE(x.has_value());
            ASSERT_TRUE(y.has_value());
            x_samples[sample] = *x;
            y_samples[sample] = *y;
        }

        for (auto x_bits = std::uint32_t{0}; x_bits <= partition_bits; ++x_bits) {
            const auto y_bits = partition_bits - x_bits;
            const auto y_bin_count = std::size_t{1} << y_bits;
            std::vector<std::uint16_t> histogram(cell_count, 0);
            for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
                const auto x_bin = leading_bin(x_samples[sample], x_bits);
                const auto y_bin = leading_bin(y_samples[sample], y_bits);
                ++histogram[x_bin * y_bin_count + y_bin];
            }
            EXPECT_TRUE(std::ranges::all_of(histogram, [expected_per_cell](const auto count) {
                return count == expected_per_cell;
            }));
        }
    }
}

TEST(OwenScramblingTest, DefinesAnExplicitReproducibleHostContract) {
    static_assert(
        std::is_same_v<decltype(owen_scrambled_sobol_bits({}, 0)), core::Result<std::uint64_t>>);
    static_assert(std::is_same_v<decltype(owen_scrambled_sobol_1d<TransportScalar>({}, 0)),
                                 core::Result<TransportScalar>>);
    static_assert(std::is_same_v<decltype(owen_scrambled_sobol_1d<ReferenceScalar>({}, 0)),
                                 core::Result<ReferenceScalar>>);

    const auto raw = sobol_sample_bits(0, 0);
    const auto scrambled = owen_scrambled_sobol_bits({}, 0);
    ASSERT_TRUE(raw.has_value());
    ASSERT_TRUE(scrambled.has_value());
    EXPECT_EQ(*raw, 0U);
    EXPECT_NE(*scrambled, *raw);

    constexpr auto shared_raw_index = SampleStreamIndex{.sample_index = 1};
    const auto dimension_zero = owen_scrambled_sobol_bits(shared_raw_index, 0);
    const auto dimension_one = owen_scrambled_sobol_bits(shared_raw_index, 1);
    const auto dimension_last =
        owen_scrambled_sobol_bits(shared_raw_index, SobolDimensionCount - 1);
    ASSERT_TRUE(dimension_zero.has_value());
    ASSERT_TRUE(dimension_one.has_value());
    ASSERT_TRUE(dimension_last.has_value());
    EXPECT_NE(*dimension_zero, *dimension_one);
    EXPECT_NE(*dimension_zero, *dimension_last);
    EXPECT_NE(*dimension_one, *dimension_last);
}

TEST(OwenScramblingTest, MatchesFrozenNestedBinaryScrambleVectors) {
    for (const auto& expected : known_owen_bits) {
        const auto bits = owen_scrambled_sobol_bits(expected.index, expected.dimension);
        ASSERT_TRUE(bits.has_value());
        EXPECT_EQ(*bits, expected.bits);
    }
}

TEST(OwenScramblingTest, ConvertsOneCanonicalWordToBothPrecisions) {
    for (const auto& expected : known_owen_bits) {
        const auto transport =
            owen_scrambled_sobol_1d<TransportScalar>(expected.index, expected.dimension);
        const auto reference =
            owen_scrambled_sobol_1d<ReferenceScalar>(expected.index, expected.dimension);
        ASSERT_TRUE(transport.has_value());
        ASSERT_TRUE(reference.has_value());

        const auto transport_numerator = static_cast<std::uint32_t>(expected.bits >> 40U);
        const auto reference_numerator = expected.bits >> 11U;
        EXPECT_EQ(*transport,
                  static_cast<TransportScalar>(transport_numerator) * TransportScalar{0x1p-24F});
        EXPECT_EQ(*reference,
                  static_cast<ReferenceScalar>(reference_numerator) * ReferenceScalar{0x1p-53});
        EXPECT_GE(*transport, TransportScalar{0});
        EXPECT_LT(*transport, TransportScalar{1});
        EXPECT_GE(*reference, ReferenceScalar{0});
        EXPECT_LT(*reference, ReferenceScalar{1});
    }
}

TEST(OwenScramblingTest, ReplaysInReverseOrderAndSeparatesEveryTreeAddressField) {
    constexpr auto sample_count = std::size_t{512};
    constexpr auto base_index = SampleStreamIndex{
        .pixel_x = 113,
        .pixel_y = 47,
        .seed = 0xA5A5F00D12345678ULL,
    };
    std::array<std::uint64_t, sample_count> recorded{};

    for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
        auto index = base_index;
        index.sample_index = sample;
        const auto bits = owen_scrambled_sobol_bits(index, 63);
        ASSERT_TRUE(bits.has_value());
        recorded[sample] = *bits;

        auto interleaved = index;
        interleaved.pixel_x ^= 0x80000000U;
        ASSERT_TRUE(owen_scrambled_sobol_bits(interleaved, 21'200).has_value());
    }

    for (auto reverse = sample_count; reverse > 0; --reverse) {
        const auto sample = reverse - 1;
        auto index = base_index;
        index.sample_index = sample;
        const auto bits = owen_scrambled_sobol_bits(index, 63);
        ASSERT_TRUE(bits.has_value());
        EXPECT_EQ(*bits, recorded[sample]);
    }

    for (const auto variant : std::array{
             SampleStreamIndex{.pixel_x = base_index.pixel_x ^ 0x80000000U,
                               .pixel_y = base_index.pixel_y,
                               .seed = base_index.seed},
             SampleStreamIndex{.pixel_x = base_index.pixel_x,
                               .pixel_y = base_index.pixel_y ^ 0x80000000U,
                               .seed = base_index.seed},
             SampleStreamIndex{.pixel_x = base_index.pixel_x,
                               .pixel_y = base_index.pixel_y,
                               .seed = base_index.seed ^ 0x8000000000000000ULL},
         }) {
        auto changed = std::array<std::uint64_t, sample_count>{};
        for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
            auto index = variant;
            index.sample_index = sample;
            const auto bits = owen_scrambled_sobol_bits(index, 63);
            ASSERT_TRUE(bits.has_value());
            changed[sample] = *bits;
        }
        EXPECT_NE(changed, recorded);
    }
}

TEST(OwenScramblingTest, RejectsCoordinatesOutsideSobolWithoutSubstitution) {
    constexpr auto index = SampleStreamIndex{
        .pixel_x = std::numeric_limits<std::uint32_t>::max(),
        .pixel_y = std::numeric_limits<std::uint32_t>::max(),
        .sample_index = std::numeric_limits<std::uint64_t>::max(),
        .seed = std::numeric_limits<std::uint64_t>::max(),
    };
    ASSERT_TRUE(owen_scrambled_sobol_bits(index, SobolDimensionCount - 1).has_value());

    for (const auto dimension :
         std::array{SobolDimensionCount, std::numeric_limits<std::uint64_t>::max()}) {
        const auto bits = owen_scrambled_sobol_bits(index, dimension);
        const auto transport = owen_scrambled_sobol_1d<TransportScalar>(index, dimension);
        const auto reference = owen_scrambled_sobol_1d<ReferenceScalar>(index, dimension);
        ASSERT_FALSE(bits.has_value());
        ASSERT_FALSE(transport.has_value());
        ASSERT_FALSE(reference.has_value());
        EXPECT_EQ(bits.error().code, core::StatusCode::resource_exhausted);
        EXPECT_EQ(transport.error().code, bits.error().code);
        EXPECT_EQ(reference.error().code, bits.error().code);
        EXPECT_EQ(bits.error().message, "Sobol sampling supports dimensions [0, 21200].");
        EXPECT_EQ(transport.error().message, bits.error().message);
        EXPECT_EQ(reference.error().message, bits.error().message);
    }
}

TEST(OwenScramblingTest, PreservesEverySelectedDyadicPermutation) {
    constexpr auto sample_count = std::size_t{4'096};
    constexpr auto dimensions = std::array<std::uint64_t, 6>{
        0, 1, 63, 1'023, 16'383, 21'200,
    };
    constexpr auto addresses = std::array{
        SampleStreamIndex{},
        SampleStreamIndex{.pixel_x = 113, .pixel_y = 47, .seed = 0xA5A5F00D12345678ULL},
    };
    constexpr auto block_offsets = std::array<std::uint64_t, 2>{0, 3 * sample_count};

    for (const auto base_index : addresses) {
        for (const auto dimension : dimensions) {
            for (const auto block_offset : block_offsets) {
                std::array<std::uint16_t, sample_count> histogram{};
                for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
                    auto index = base_index;
                    index.sample_index = block_offset + static_cast<std::uint64_t>(sample);
                    const auto bits = owen_scrambled_sobol_bits(index, dimension);
                    ASSERT_TRUE(bits.has_value());
                    const auto bin = leading_bin(*bits, 12);
                    ASSERT_LT(bin, histogram.size());
                    ++histogram[bin];
                }
                EXPECT_TRUE(
                    std::ranges::all_of(histogram, [](const auto count) { return count == 1; }));
            }
        }
    }
}

TEST(OwenScramblingTest, PreservesLowAndHighDimensionalProjectionNets) {
    constexpr auto low_blocks = std::array<std::uint64_t, 2>{0, 3 * 4'096};
    expect_exact_owen_projection(
        SampleStreamIndex{.pixel_x = 113, .pixel_y = 47, .seed = 0xA5A5F00D12345678ULL}, 0, 1,
        low_blocks, 12, 1);

    constexpr auto high_blocks =
        std::array<std::uint64_t, 4>{0, 4'096, 3 * 4'096, std::uint64_t{1} << 20U};
    expect_exact_owen_projection(
        SampleStreamIndex{.pixel_x = 17, .pixel_y = 29, .seed = 0x0123456789ABCDEFULL}, 16'383,
        21'200, high_blocks, 10, 4);
}

inline constexpr auto VisualizerExtent = std::uint32_t{256};
inline constexpr auto VisualizerSampleCount = std::size_t{4'096};
inline constexpr auto VisualizerIndex = SampleStreamIndex{
    .pixel_x = 113,
    .pixel_y = 47,
    .seed = 0xA5A5F00D12345678ULL,
};

struct VisualizerData final {
    std::vector<std::uint16_t> pixels;
    std::array<std::uint16_t, 16> local_positions{};
    std::uint16_t maximum_toroidal_line_count{};
};

[[nodiscard]] std::optional<VisualizerData> make_visualizer_data() {
    auto data = VisualizerData{
        .pixels = std::vector<std::uint16_t>(
            static_cast<std::size_t>(VisualizerExtent) * VisualizerExtent, 0),
    };
    std::array<std::array<std::uint16_t, VisualizerExtent>, VisualizerExtent> line_histograms{};

    for (auto sample = std::size_t{0}; sample < VisualizerSampleCount; ++sample) {
        auto index = VisualizerIndex;
        index.sample_index = sample;
        const auto x_bits = owen_scrambled_sobol_bits(index, 0);
        const auto y_bits = owen_scrambled_sobol_bits(index, 1);
        if (!x_bits.has_value() || !y_bits.has_value()) {
            return std::nullopt;
        }

        const auto x = static_cast<std::uint32_t>(*x_bits >> 56U);
        const auto y = static_cast<std::uint32_t>(*y_bits >> 56U);
        ++data.pixels[static_cast<std::size_t>(y) * VisualizerExtent + x];
        ++data.local_positions[(y & 3U) * 4U + (x & 3U)];

        for (auto slope = std::uint32_t{0}; slope < VisualizerExtent; ++slope) {
            const auto intercept = (y - slope * x) & 255U;
            ++line_histograms[slope][intercept];
        }
    }

    for (const auto& slope : line_histograms) {
        data.maximum_toroidal_line_count =
            std::max(data.maximum_toroidal_line_count, *std::ranges::max_element(slope));
    }
    return data;
}

TEST(OwenScramblingTest, AvoidsCoarseVisualizerPatterns) {
    const auto data = make_visualizer_data();
    ASSERT_TRUE(data.has_value());

    std::array<std::uint16_t, VisualizerExtent> row_counts{};
    std::array<std::uint16_t, VisualizerExtent> column_counts{};
    std::array<std::uint16_t, (VisualizerExtent / 4U) * (VisualizerExtent / 4U)> macro_counts{};
    std::array<std::uint16_t, (VisualizerExtent / 16U) * (VisualizerExtent / 16U)> tile_counts{};
    auto occupied_count = std::size_t{0};
    for (auto y = std::uint32_t{0}; y < VisualizerExtent; ++y) {
        for (auto x = std::uint32_t{0}; x < VisualizerExtent; ++x) {
            const auto count = data->pixels[static_cast<std::size_t>(y) * VisualizerExtent + x];
            if (count != 0) {
                ++occupied_count;
                row_counts[y] += count;
                column_counts[x] += count;
                ++macro_counts[(y / 4U) * (VisualizerExtent / 4U) + x / 4U];
                ++tile_counts[(y / 16U) * (VisualizerExtent / 16U) + x / 16U];
            }
            EXPECT_LE(count, 1);
        }
    }

    EXPECT_EQ(occupied_count, VisualizerSampleCount);
    EXPECT_TRUE(std::ranges::all_of(row_counts, [](const auto count) { return count == 16; }));
    EXPECT_TRUE(std::ranges::all_of(column_counts, [](const auto count) { return count == 16; }));
    EXPECT_TRUE(std::ranges::all_of(macro_counts, [](const auto count) { return count == 1; }));
    EXPECT_TRUE(std::ranges::all_of(tile_counts, [](const auto count) { return count == 16; }));
    for (const auto count : data->local_positions) {
        EXPECT_GE(count, 192);
        EXPECT_LE(count, 320);
    }
    EXPECT_LE(data->maximum_toroidal_line_count, 64);
}

#if BLACKFRAME_HAS_PNG_PREVIEW

[[nodiscard]] std::optional<std::filesystem::path> visualizer_checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1 ? std::optional{std::filesystem::path{value}} : std::nullopt;
    std::free(value);
    return path;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path{value};
#endif
}

[[nodiscard]] std::filesystem::path visualizer_artifact_path() {
    if (const auto checksum_output = visualizer_checksum_output_path();
        checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_RENDERER_TEST_OUTPUT_DIR} /
           "owen-scrambling-visualizer.png";
}

TEST(OwenScramblingTest, WritesStableVisualizer) {
    const auto data = make_visualizer_data();
    ASSERT_TRUE(data.has_value());

    auto film = Film::create(RenderExtent{.width = VisualizerExtent, .height = VisualizerExtent});
    ASSERT_TRUE(film.has_value());
    for (auto y = std::uint32_t{0}; y < VisualizerExtent; ++y) {
        for (auto x = std::uint32_t{0}; x < VisualizerExtent; ++x) {
            const auto occupied =
                data->pixels[static_cast<std::size_t>(y) * VisualizerExtent + x] != 0;
            const auto value = occupied ? TransportScalar{1} : TransportScalar{0};
            ASSERT_TRUE(
                film->add_sample(x, y, LinearRGB{.red = value, .green = value, .blue = value}, 1.0F)
                    .has_value());
        }
    }

    const auto output_path = visualizer_artifact_path();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_TRUE(write_png_preview(*film, output_path).has_value());
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(width, static_cast<int>(VisualizerExtent));
    EXPECT_EQ(height, static_cast<int>(VisualizerExtent));
    EXPECT_EQ(components, 3);

    auto white_pixels = std::size_t{0};
    for (auto pixel = std::size_t{0}; pixel < data->pixels.size(); ++pixel) {
        const auto red = decoded[pixel * 3];
        const auto green = decoded[pixel * 3 + 1];
        const auto blue = decoded[pixel * 3 + 2];
        EXPECT_EQ(red, green);
        EXPECT_EQ(green, blue);
        EXPECT_TRUE(red == 0 || red == 255);
        white_pixels += red == 255 ? 1U : 0U;
    }
    EXPECT_EQ(white_pixels, VisualizerSampleCount);
    stbi_image_free(decoded);

    if (!visualizer_checksum_output_path().has_value()) {
        EXPECT_TRUE(std::filesystem::remove(output_path));
    }
}

#else

TEST(OwenScramblingTest, WritesStableVisualizer) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
