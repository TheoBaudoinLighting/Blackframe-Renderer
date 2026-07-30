#include <Blackframe/Renderer/SobolSampling.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

struct KnownSobolBits final {
    std::uint64_t sample_index;
    std::uint64_t dimension;
    std::uint64_t bits;
};

constexpr auto published_first_ten_numerators = std::array{
    std::array<std::uint64_t, 3>{0, 0, 0},   std::array<std::uint64_t, 3>{8, 8, 8},
    std::array<std::uint64_t, 3>{12, 4, 4},  std::array<std::uint64_t, 3>{4, 12, 12},
    std::array<std::uint64_t, 3>{6, 6, 10},  std::array<std::uint64_t, 3>{14, 14, 2},
    std::array<std::uint64_t, 3>{10, 2, 14}, std::array<std::uint64_t, 3>{2, 10, 6},
    std::array<std::uint64_t, 3>{3, 5, 15},  std::array<std::uint64_t, 3>{11, 13, 7},
};

constexpr auto known_high_dimensional_bits = std::array{
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 63, .bits = 0x013E8C6349603C80ULL},
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 255, .bits = 0xB321991D6689DC80ULL},
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 1'023, .bits = 0xE4BCA0B1ED509A80ULL},
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 4'095, .bits = 0x572DD6D8275D9480ULL},
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 16'383, .bits = 0x66A5274EF5FA7480ULL},
    KnownSobolBits{
        .sample_index = 0x0123456789ABCDEFULL, .dimension = 21'200, .bits = 0xF4C1FC94854F3780ULL},
    KnownSobolBits{
        .sample_index = 0x8000000000000000ULL, .dimension = 21'200, .bits = 0x45AC83CC9F734085ULL},
    KnownSobolBits{.sample_index = std::numeric_limits<std::uint64_t>::max(),
                   .dimension = 21'200,
                   .bits = 0xA8D3F37AB495C0C3ULL},
    KnownSobolBits{.sample_index = std::numeric_limits<std::uint64_t>::max(),
                   .dimension = 1,
                   .bits = std::numeric_limits<std::uint64_t>::max()},
};

[[nodiscard]] constexpr std::size_t leading_bin(const std::uint64_t bits,
                                                const std::uint32_t bin_bits) noexcept {
    if (bin_bits == 0) {
        return 0;
    }
    return static_cast<std::size_t>(bits >> (64U - bin_bits));
}

void expect_exact_projection(const std::uint64_t x_dimension, const std::uint64_t y_dimension,
                             const std::span<const std::uint64_t> block_offsets,
                             const std::uint32_t partition_bits,
                             const std::uint16_t expected_per_cell) {
    constexpr auto sample_count = std::size_t{4'096};
    const auto cell_count = std::size_t{1} << partition_bits;

    for (const auto block_offset : block_offsets) {
        std::vector<std::uint64_t> x_samples(sample_count);
        std::vector<std::uint64_t> y_samples(sample_count);
        for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
            const auto sample_index = block_offset + static_cast<std::uint64_t>(sample);
            const auto x = sobol_sample_bits(sample_index, x_dimension);
            const auto y = sobol_sample_bits(sample_index, y_dimension);
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

TEST(SobolSamplingTest, DefinesTheCompleteJoeKuoDomainAndPrecisionContract) {
    static_assert(SobolDimensionCount == 21'201);
    static_assert(SobolDirectionBitCount == 64);
    static_assert(std::is_same_v<decltype(sobol_sample_bits(0, 0)), core::Result<std::uint64_t>>);
    static_assert(std::is_same_v<decltype(sobol_sample_1d<TransportScalar>(0, 0)),
                                 core::Result<TransportScalar>>);
    static_assert(std::is_same_v<decltype(sobol_sample_1d<ReferenceScalar>(0, 0)),
                                 core::Result<ReferenceScalar>>);

    for (const auto dimension : std::array{
             std::uint64_t{0},
             std::uint64_t{1},
             SobolDimensionCount - 1,
         }) {
        const auto bits = sobol_sample_bits(0, dimension);
        const auto transport = sobol_sample_1d<TransportScalar>(0, dimension);
        const auto reference = sobol_sample_1d<ReferenceScalar>(0, dimension);
        ASSERT_TRUE(bits.has_value());
        ASSERT_TRUE(transport.has_value());
        ASSERT_TRUE(reference.has_value());
        EXPECT_EQ(*bits, 0U);
        EXPECT_EQ(*transport, 0.0F);
        EXPECT_EQ(*reference, 0.0);
    }
}

TEST(SobolSamplingTest, MatchesThePublishedFirstTenPointsInThreeDimensions) {
    for (auto sample = std::size_t{0}; sample < published_first_ten_numerators.size(); ++sample) {
        for (auto dimension = std::size_t{0}; dimension < 3; ++dimension) {
            const auto bits = sobol_sample_bits(sample, dimension);
            ASSERT_TRUE(bits.has_value());
            EXPECT_EQ(*bits, published_first_ten_numerators[sample][dimension] << 60U);
        }
    }
}

TEST(SobolSamplingTest, MatchesKnownAnswersAcrossTheFullTableAndIndexWidth) {
    for (const auto& expected : known_high_dimensional_bits) {
        const auto bits = sobol_sample_bits(expected.sample_index, expected.dimension);
        ASSERT_TRUE(bits.has_value());
        EXPECT_EQ(*bits, expected.bits);
    }
}

TEST(SobolSamplingTest, ConvertsTheSameBitsToExactTransportAndReferenceValues) {
    for (const auto& expected : known_high_dimensional_bits) {
        const auto transport =
            sobol_sample_1d<TransportScalar>(expected.sample_index, expected.dimension);
        const auto reference =
            sobol_sample_1d<ReferenceScalar>(expected.sample_index, expected.dimension);
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

TEST(SobolSamplingTest, RejectsEveryCoordinateOutsideTheDirectionTable) {
    const auto last = sobol_sample_bits(37, SobolDimensionCount - 1);
    ASSERT_TRUE(last.has_value());

    for (const auto dimension :
         std::array{SobolDimensionCount, std::numeric_limits<std::uint64_t>::max()}) {
        const auto bits = sobol_sample_bits(37, dimension);
        const auto transport = sobol_sample_1d<TransportScalar>(37, dimension);
        const auto reference = sobol_sample_1d<ReferenceScalar>(37, dimension);
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

TEST(SobolSamplingTest, CoversEveryDyadicBinInLowAndHighDimensions) {
    constexpr auto sample_count = std::size_t{4'096};
    constexpr auto bin_shift = std::uint32_t{52};
    constexpr auto dimensions = std::array<std::uint64_t, 8>{
        0, 1, 63, 255, 1'023, 4'095, 16'383, 21'200,
    };
    constexpr auto block_offsets = std::array<std::uint64_t, 2>{
        0,
        3 * sample_count,
    };

    for (const auto dimension : dimensions) {
        for (const auto block_offset : block_offsets) {
            std::array<std::uint16_t, sample_count> histogram{};
            for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
                const auto bits =
                    sobol_sample_bits(block_offset + static_cast<std::uint64_t>(sample), dimension);
                ASSERT_TRUE(bits.has_value());
                const auto bin = static_cast<std::size_t>(*bits >> bin_shift);
                ASSERT_LT(bin, histogram.size());
                ++histogram[bin];
            }
            EXPECT_TRUE(
                std::ranges::all_of(histogram, [](const auto count) { return count == 1; }));
        }
    }
}

TEST(SobolSamplingTest, FormsExactDyadicNetsInLowAndHighDimensionalProjections) {
    constexpr auto low_blocks = std::array<std::uint64_t, 2>{0, 3 * 4'096};
    expect_exact_projection(0, 1, low_blocks, 12, 1);

    constexpr auto high_blocks =
        std::array<std::uint64_t, 4>{0, 4'096, 3 * 4'096, std::uint64_t{1} << 20U};
    expect_exact_projection(16'383, 21'200, high_blocks, 10, 4);
}

TEST(SobolSamplingTest, BoundsHighDimensionalGridStarDiscrepancy) {
    constexpr auto sample_count = std::size_t{4'096};
    constexpr auto grid_size = std::size_t{64};
    constexpr auto grid_area = grid_size * grid_size;
    std::array<std::array<std::uint16_t, grid_size>, grid_size> histogram{};

    for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
        const auto x = sobol_sample_bits(sample, 16'383);
        const auto y = sobol_sample_bits(sample, 21'200);
        ASSERT_TRUE(x.has_value());
        ASSERT_TRUE(y.has_value());
        const auto x_bin = leading_bin(*x, 6);
        const auto y_bin = leading_bin(*y, 6);
        ++histogram[y_bin][x_bin];
    }

    std::array<std::array<std::uint32_t, grid_size + 1>, grid_size + 1> prefix{};
    for (auto y = std::size_t{1}; y <= grid_size; ++y) {
        for (auto x = std::size_t{1}; x <= grid_size; ++x) {
            prefix[y][x] = histogram[y - 1][x - 1] + prefix[y - 1][x] + prefix[y][x - 1] -
                           prefix[y - 1][x - 1];
        }
    }

    auto maximum_numerator = std::uint64_t{0};
    for (auto y = std::size_t{0}; y <= grid_size; ++y) {
        for (auto x = std::size_t{0}; x <= grid_size; ++x) {
            const auto empirical = static_cast<std::uint64_t>(prefix[y][x]) * grid_area;
            const auto expected = static_cast<std::uint64_t>(sample_count) * x * y;
            const auto difference =
                empirical > expected ? empirical - expected : expected - empirical;
            maximum_numerator = std::max(maximum_numerator, difference);
        }
    }

    constexpr auto discrepancy_denominator = sample_count * grid_area;
    static_assert(discrepancy_denominator == 16'777'216);
    EXPECT_EQ(maximum_numerator, 4'096U);
    EXPECT_LE(maximum_numerator, discrepancy_denominator / 1'024U);
}

} // namespace
} // namespace blackframe::renderer
