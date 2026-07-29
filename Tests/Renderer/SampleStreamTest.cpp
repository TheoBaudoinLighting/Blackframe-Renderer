#include <Blackframe/Renderer/SampleStream.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

struct KnownSample final {
    SampleStreamIndex index;
    std::uint64_t dimension;
    std::uint32_t transport_numerator;
    std::uint64_t reference_numerator;
};

constexpr auto known_samples = std::array{
    KnownSample{
        .index = {},
        .dimension = 0,
        .transport_numerator = 0x2F37B1U,
        .reference_numerator = 0x05E6F6339B5611ULL,
    },
    KnownSample{
        .index = {.pixel_x = 17, .pixel_y = 29, .sample_index = 0, .seed = 0x0123456789ABCDEFULL},
        .dimension = 0,
        .transport_numerator = 0xDE27E5U,
        .reference_numerator = 0x1BC4FCA4264791ULL,
    },
    KnownSample{
        .index =
            {.pixel_x = 17, .pixel_y = 29, .sample_index = 4095, .seed = 0x0123456789ABCDEFULL},
        .dimension = 1,
        .transport_numerator = 0xDC4661U,
        .reference_numerator = 0x1B88CC28476D40ULL,
    },
    KnownSample{
        .index = {.pixel_x = std::numeric_limits<std::uint32_t>::max(),
                  .pixel_y = std::numeric_limits<std::uint32_t>::max(),
                  .sample_index = std::numeric_limits<std::uint64_t>::max(),
                  .seed = std::numeric_limits<std::uint64_t>::max()},
        .dimension = std::numeric_limits<std::uint64_t>::max(),
        .transport_numerator = 0x8F4FD2U,
        .reference_numerator = 0x11E9FA55578958ULL,
    },
};

TEST(SampleStreamTest, DefinesDistinctTrivialTransportAndReferenceStreams) {
    static_assert(std::is_same_v<SampleStream::value_type, TransportScalar>);
    static_assert(std::is_same_v<ReferenceSampleStream::value_type, ReferenceScalar>);
    static_assert(!std::is_same_v<SampleStream, ReferenceSampleStream>);
    static_assert(std::is_standard_layout_v<SampleStreamIndex>);
    static_assert(std::is_trivially_copyable_v<SampleStreamIndex>);
    static_assert(std::is_standard_layout_v<SampleStream>);
    static_assert(std::is_trivially_copyable_v<SampleStream>);
    static_assert(std::is_standard_layout_v<ReferenceSampleStream>);
    static_assert(std::is_trivially_copyable_v<ReferenceSampleStream>);

    constexpr auto index = SampleStreamIndex{
        .pixel_x = 7,
        .pixel_y = 11,
        .sample_index = 13,
        .seed = 17,
    };
    constexpr auto stream = SampleStream{index};
    static_assert(stream.index() == index);
    EXPECT_EQ(stream.index(), index);
}

TEST(SampleStreamTest, MatchesFixedKnownAnswerValues) {
    for (const auto& expected : known_samples) {
        const auto transport = SampleStream{expected.index}.sample_1d(expected.dimension);
        const auto reference = ReferenceSampleStream{expected.index}.sample_1d(expected.dimension);

        EXPECT_EQ(transport, static_cast<TransportScalar>(expected.transport_numerator) * 0x1p-24F);
        EXPECT_EQ(reference, static_cast<ReferenceScalar>(expected.reference_numerator) * 0x1p-53);
    }
}

TEST(SampleStreamTest, RandomDimensionAccessIsStableAndOrderIndependent) {
    constexpr auto dimension_count = std::size_t{4096};
    constexpr auto index = SampleStreamIndex{
        .pixel_x = 113,
        .pixel_y = 47,
        .sample_index = 91,
        .seed = 0xA5A5F00D12345678ULL,
    };
    const auto stream = SampleStream{index};
    std::array<TransportScalar, dimension_count> forward{};

    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
        forward[dimension] = stream.sample_1d(static_cast<std::uint64_t>(dimension));
    }

    for (std::size_t reverse = dimension_count; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        EXPECT_EQ(stream.sample_1d(static_cast<std::uint64_t>(dimension)), forward[dimension]);
    }

    const auto replay = SampleStream{index};
    for (std::size_t order = 0; order < dimension_count; ++order) {
        const auto dimension = (order * 4051U + 17U) % dimension_count;
        EXPECT_EQ(replay.sample_1d(static_cast<std::uint64_t>(dimension)), forward[dimension]);
    }
}

TEST(SampleStreamTest, InterleavedStreamsDoNotShareGeneratorState) {
    constexpr auto first_index = SampleStreamIndex{
        .pixel_x = 1,
        .pixel_y = 2,
        .sample_index = 3,
        .seed = 4,
    };
    constexpr auto second_index = SampleStreamIndex{
        .pixel_x = 5,
        .pixel_y = 6,
        .sample_index = 7,
        .seed = 8,
    };
    constexpr auto dimension_count = std::size_t{256};
    const auto first = ReferenceSampleStream{first_index};
    const auto second = ReferenceSampleStream{second_index};
    std::array<ReferenceScalar, dimension_count> recorded{};

    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
        recorded[dimension] = first.sample_1d(static_cast<std::uint64_t>(dimension));
        static_cast<void>(
            second.sample_1d(static_cast<std::uint64_t>(dimension_count - dimension)));
    }

    const auto replay = ReferenceSampleStream{first_index};
    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
        EXPECT_EQ(replay.sample_1d(static_cast<std::uint64_t>(dimension)), recorded[dimension]);
    }
}

TEST(SampleStreamTest, EveryAddressComponentUsesItsFullDeclaredWidth) {
    constexpr auto base_index = SampleStreamIndex{
        .pixel_x = 0x01234567U,
        .pixel_y = 0x89ABCDEFU,
        .sample_index = 0x0123456789ABCDEFULL,
        .seed = 0xFEDCBA9876543210ULL,
    };
    constexpr auto dimension = 0x8000000000000025ULL;
    const auto baseline = ReferenceSampleStream{base_index}.sample_1d(dimension);

    auto changed = base_index;
    changed.pixel_x ^= 0x80000000U;
    EXPECT_NE(ReferenceSampleStream{changed}.sample_1d(dimension), baseline);

    changed = base_index;
    changed.pixel_y ^= 0x80000000U;
    EXPECT_NE(ReferenceSampleStream{changed}.sample_1d(dimension), baseline);

    changed = base_index;
    changed.sample_index ^= 0x8000000000000000ULL;
    EXPECT_NE(ReferenceSampleStream{changed}.sample_1d(dimension), baseline);

    changed = base_index;
    changed.seed ^= 0x8000000000000000ULL;
    EXPECT_NE(ReferenceSampleStream{changed}.sample_1d(dimension), baseline);

    EXPECT_NE(ReferenceSampleStream{base_index}.sample_1d(dimension ^ 0x8000000000000000ULL),
              baseline);
}

TEST(SampleStreamTest, PrecisionConversionsAreExactAndRemainInTheUnitInterval) {
    constexpr auto index = SampleStreamIndex{
        .pixel_x = std::numeric_limits<std::uint32_t>::max(),
        .pixel_y = std::numeric_limits<std::uint32_t>::max(),
        .sample_index = std::numeric_limits<std::uint64_t>::max(),
        .seed = std::numeric_limits<std::uint64_t>::max(),
    };
    const auto transport = SampleStream{index};
    const auto reference = ReferenceSampleStream{index};

    for (const auto dimension : std::array{
             std::uint64_t{0},
             std::uint64_t{1},
             std::uint64_t{37},
             std::uint64_t{1} << 32U,
             std::numeric_limits<std::uint64_t>::max(),
         }) {
        const auto transport_value = transport.sample_1d(dimension);
        const auto reference_value = reference.sample_1d(dimension);

        EXPECT_GE(transport_value, TransportScalar{0});
        EXPECT_LT(transport_value, TransportScalar{1});
        EXPECT_GE(reference_value, ReferenceScalar{0});
        EXPECT_LT(reference_value, ReferenceScalar{1});
        EXPECT_EQ(std::trunc(transport_value * 0x1p24F), transport_value * 0x1p24F);
        EXPECT_EQ(std::trunc(reference_value * 0x1p53), reference_value * 0x1p53);

        const auto shared_numerator =
            static_cast<std::uint32_t>(reference_value * ReferenceScalar{0x1p24});
        EXPECT_EQ(transport_value,
                  static_cast<TransportScalar>(shared_numerator) * TransportScalar{0x1p-24F});
    }
}

} // namespace
} // namespace blackframe::renderer
