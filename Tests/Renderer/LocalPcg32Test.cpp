#include <Blackframe/Renderer/LocalPcg32.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

struct KnownLocalSequence final {
    SampleStreamIndex path;
    std::uint64_t anchor_dimension;
    std::array<std::uint32_t, 8> expected;
};

constexpr auto known_sequences = std::array{
    KnownLocalSequence{
        .path = {},
        .anchor_dimension = 0,
        .expected = {0x6C865DF7U, 0x3E844330U, 0xFA8EF026U, 0x08EB65AFU, 0xC3EBB507U, 0xEE7087A3U,
                     0xA93C7A98U, 0x286A7955U},
    },
    KnownLocalSequence{
        .path =
            {
                .pixel_x = 17,
                .pixel_y = 29,
                .sample_index = 4095,
                .seed = 0x0123456789ABCDEFULL,
            },
        .anchor_dimension = 13,
        .expected = {0xCBCDF7C4U, 0x78A6929FU, 0xED2AA932U, 0x91BCD72EU, 0x16D00362U, 0x61CBF6ADU,
                     0x04A66931U, 0x9A26CCE8U},
    },
    KnownLocalSequence{
        .path =
            {
                .pixel_x = std::numeric_limits<std::uint32_t>::max(),
                .pixel_y = std::numeric_limits<std::uint32_t>::max(),
                .sample_index = std::numeric_limits<std::uint64_t>::max(),
                .seed = std::numeric_limits<std::uint64_t>::max(),
            },
        .anchor_dimension = std::numeric_limits<std::uint64_t>::max(),
        .expected = {0x7FDB846CU, 0x61375EA0U, 0xD94F5E8DU, 0xDCA8CBBFU, 0xBFFD76BFU, 0x96BCA264U,
                     0x2B81F990U, 0x53F1C368U},
    },
};

[[nodiscard]] std::uint64_t consume_variable_loop(LocalPcg32& rng) {
    const auto draw_count = rng.next_u32() & 7U;
    for (std::uint32_t draw = 0; draw < draw_count; ++draw) {
        static_cast<void>(rng.next_u32());
    }
    return rng.next_u64();
}

TEST(LocalPcg32Test, DefinesExplicitTrivialLocalState) {
    static_assert(!std::is_default_constructible_v<LocalPcg32>);
    static_assert(std::is_nothrow_constructible_v<LocalPcg32, SampleStreamIndex, std::uint64_t>);
    static_assert(std::is_standard_layout_v<LocalPcg32>);
    static_assert(std::is_trivially_copyable_v<LocalPcg32>);
    static_assert(sizeof(LocalPcg32) == 2 * sizeof(std::uint64_t));

    SUCCEED();
}

TEST(LocalPcg32Test, MatchesFrozenPathAddressSequences) {
    for (const auto& known : known_sequences) {
        auto rng = LocalPcg32{known.path, known.anchor_dimension};
        for (const auto expected : known.expected) {
            EXPECT_EQ(rng.next_u32(), expected);
        }
    }
}

TEST(LocalPcg32Test, AssemblesStableHighThenLowWords) {
    auto rng = LocalPcg32{{}, 0};
    constexpr auto expected = std::array{
        std::uint64_t{0x6C865DF73E844330ULL},
        std::uint64_t{0xFA8EF02608EB65AFULL},
        std::uint64_t{0xC3EBB507EE7087A3ULL},
        std::uint64_t{0xA93C7A98286A7955ULL},
    };

    for (const auto value : expected) {
        EXPECT_EQ(rng.next_u64(), value);
    }
}

TEST(LocalPcg32Test, ReplaysVariableConsumptionBitForBit) {
    constexpr auto path = SampleStreamIndex{
        .pixel_x = 113,
        .pixel_y = 47,
        .sample_index = 91,
        .seed = 0xA5A5F00D12345678ULL,
    };
    constexpr auto event_count = std::size_t{2048};
    std::array<std::uint64_t, event_count> recorded{};
    auto first = LocalPcg32{path, 13};

    for (auto& value : recorded) {
        value = consume_variable_loop(first);
    }

    auto replay = LocalPcg32{path, 13};
    for (const auto expected : recorded) {
        EXPECT_EQ(consume_variable_loop(replay), expected);
    }
}

TEST(LocalPcg32Test, InterleavedPathsKeepIndependentConsumptionState) {
    constexpr auto first_path = SampleStreamIndex{
        .pixel_x = 1,
        .pixel_y = 2,
        .sample_index = 3,
        .seed = 4,
    };
    constexpr auto second_path = SampleStreamIndex{
        .pixel_x = 5,
        .pixel_y = 6,
        .sample_index = 7,
        .seed = 8,
    };
    constexpr auto event_count = std::size_t{1024};
    std::array<std::uint64_t, event_count> isolated_first{};
    std::array<std::uint64_t, event_count> isolated_second{};
    auto first = LocalPcg32{first_path, 10};
    auto second = LocalPcg32{second_path, 16};

    for (std::size_t event = 0; event < event_count; ++event) {
        isolated_first[event] = consume_variable_loop(first);
    }
    for (std::size_t event = 0; event < event_count; ++event) {
        isolated_second[event] = consume_variable_loop(second);
    }

    first = LocalPcg32{first_path, 10};
    second = LocalPcg32{second_path, 16};
    for (std::size_t event = 0; event < event_count; ++event) {
        if ((event & 1U) == 0U) {
            EXPECT_EQ(consume_variable_loop(first), isolated_first[event]);
            EXPECT_EQ(consume_variable_loop(second), isolated_second[event]);
        } else {
            EXPECT_EQ(consume_variable_loop(second), isolated_second[event]);
            EXPECT_EQ(consume_variable_loop(first), isolated_first[event]);
        }
    }
}

TEST(LocalPcg32Test, EveryPathAddressComponentChangesTheFrozenSequence) {
    constexpr auto base_path = SampleStreamIndex{
        .pixel_x = 0x01234567U,
        .pixel_y = 0x89ABCDEFU,
        .sample_index = 0x0123456789ABCDEFULL,
        .seed = 0xFEDCBA9876543210ULL,
    };
    constexpr auto anchor_dimension = std::uint64_t{0x8000000000000025ULL};
    struct AddressVector final {
        SampleStreamIndex path;
        std::uint64_t anchor;
        std::uint64_t expected;
    };
    constexpr auto vectors = std::array{
        AddressVector{base_path, anchor_dimension, 0x8C2DB0878EFCF93DULL},
        AddressVector{{.pixel_x = base_path.pixel_x ^ 0x80000000U,
                       .pixel_y = base_path.pixel_y,
                       .sample_index = base_path.sample_index,
                       .seed = base_path.seed},
                      anchor_dimension,
                      0x6634C80FF77B0F2FULL},
        AddressVector{{.pixel_x = base_path.pixel_x,
                       .pixel_y = base_path.pixel_y ^ 0x80000000U,
                       .sample_index = base_path.sample_index,
                       .seed = base_path.seed},
                      anchor_dimension,
                      0x3E5379885CDF321EULL},
        AddressVector{{.pixel_x = base_path.pixel_x,
                       .pixel_y = base_path.pixel_y,
                       .sample_index = base_path.sample_index ^ 0x8000000000000000ULL,
                       .seed = base_path.seed},
                      anchor_dimension,
                      0x9FDF577B0776C95BULL},
        AddressVector{{.pixel_x = base_path.pixel_x,
                       .pixel_y = base_path.pixel_y,
                       .sample_index = base_path.sample_index,
                       .seed = base_path.seed ^ 0x8000000000000000ULL},
                      anchor_dimension,
                      0x11D8C8E08CD56866ULL},
        AddressVector{base_path, anchor_dimension ^ 0x8000000000000000ULL, 0x2288C8307B129457ULL},
    };

    auto baseline = LocalPcg32{vectors.front().path, vectors.front().anchor};
    const auto baseline_value = baseline.next_u64();
    for (std::size_t index = 0; index < vectors.size(); ++index) {
        const auto& vector = vectors[index];
        auto rng = LocalPcg32{vector.path, vector.anchor};
        const auto value = rng.next_u64();
        EXPECT_EQ(value, vector.expected);
        if (index > 0) {
            EXPECT_NE(value, baseline_value) << index;
        }
    }
}

TEST(LocalPcg32Test, AValueCopyReplaysAndThenAdvancesIndependently) {
    constexpr auto path = SampleStreamIndex{
        .pixel_x = 19,
        .pixel_y = 23,
        .sample_index = 29,
        .seed = 31,
    };
    auto original = LocalPcg32{path, 37};
    for (std::size_t draw = 0; draw < 127; ++draw) {
        static_cast<void>(original.next_u32());
    }

    auto snapshot = original;
    for (std::size_t draw = 0; draw < 1024; ++draw) {
        EXPECT_EQ(original.next_u64(), snapshot.next_u64());
    }

    const auto first = original.next_u32();
    const auto second = original.next_u32();
    EXPECT_EQ(snapshot.next_u32(), first);
    EXPECT_EQ(snapshot.next_u32(), second);
}

TEST(LocalPcg32Test, TransportAndReferenceValuesConsumeTheSameWords) {
    constexpr auto path = SampleStreamIndex{
        .pixel_x = 41,
        .pixel_y = 43,
        .sample_index = 47,
        .seed = 53,
    };
    auto transport = LocalPcg32{path, 59};
    auto reference = LocalPcg32{path, 59};

    for (std::size_t draw = 0; draw < 4096; ++draw) {
        const auto transport_value = transport.next_1d<TransportScalar>();
        const auto reference_value = reference.next_1d<ReferenceScalar>();

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

    EXPECT_EQ(transport.next_u64(), reference.next_u64());
}

} // namespace
} // namespace blackframe::renderer
