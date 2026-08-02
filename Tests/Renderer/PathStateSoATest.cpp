#include <Blackframe/Renderer/PathStateSoA.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using StateFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, PathState, ReferencePathState>;

template <SpectrumScalar Scalar>
using SoAFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, PathStateSoA, ReferencePathStateSoA>;

template <SpectrumScalar Scalar> using WavelengthsFor = SampledWavelengthsT<Scalar>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
using ScalarBits =
    std::conditional_t<sizeof(Scalar) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;

template <typename SoA>
concept HasRvalueColumns = requires(SoA&& soa) { std::move(soa).columns(); };

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr ScalarBits<Scalar> scalar_bits(const Scalar value) noexcept {
    return std::bit_cast<ScalarBits<Scalar>>(value);
}

template <SpectrumScalar Scalar>
void expect_scalar_bits_equal(const Scalar actual, const Scalar expected) {
    EXPECT_EQ(scalar_bits(actual), scalar_bits(expected));
}

template <SpectrumScalar Scalar>
[[nodiscard]] WavelengthsFor<Scalar>
make_wavelengths(const std::array<Scalar, TransportSpectrumSampleCount>& nanometers,
                 const std::array<Scalar, TransportSpectrumSampleCount>& probabilities) {
    auto wavelengths = WavelengthsFor<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        wavelengths[lane] = typename WavelengthsFor<Scalar>::value_type{
            .nanometers = nanometers[lane],
            .probability =
                {
                    .value = probabilities[lane],
                    .measure = ProbabilityMeasure::wavelength,
                },
        };
    }
    return wavelengths;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::vector<StateFor<Scalar>>> make_states() {
    const auto primary_wavelengths =
        make_wavelengths<Scalar>({Scalar{360}, Scalar{470}, Scalar{600}, Scalar{830}},
                                 {Scalar{0.5}, Scalar{0.25}, Scalar{0.125}, Scalar{0.0625}});
    const auto primary = StateFor<Scalar>::create_initial(primary_wavelengths, VacuumMedium);
    if (!primary.has_value()) {
        return std::unexpected(primary.error());
    }

    const auto first_beta = SpectrumFor<Scalar>{
        .values =
            {
                -Scalar{0},
                std::numeric_limits<Scalar>::denorm_min(),
                Scalar{-1},
                std::numeric_limits<Scalar>::max(),
            },
    };
    const auto first_radiance = SpectrumFor<Scalar>{
        .values =
            {
                Scalar{0},
                -Scalar{0},
                std::numeric_limits<Scalar>::lowest(),
                Scalar{17.5},
            },
    };
    const auto first_wavelengths =
        make_wavelengths<Scalar>({Scalar{830}, Scalar{360}, Scalar{512}, Scalar{701}},
                                 {std::numeric_limits<Scalar>::denorm_min(), Scalar{0.25},
                                  Scalar{2}, std::numeric_limits<Scalar>::max()});
    const auto first = StateFor<Scalar>::create(
        first_beta, first_radiance, PathDepthCounters{.diffuse = 1, .transmission = 1},
        std::numeric_limits<Scalar>::denorm_min(), first_wavelengths,
        PathDeltaFlags::any_non_delta_bounces, MediumId{.value = 42});
    if (!first.has_value()) {
        return std::unexpected(first.error());
    }

    const auto mixed_beta = SpectrumFor<Scalar>{
        .values =
            {
                Scalar{3.25},
                -Scalar{0},
                -std::numeric_limits<Scalar>::denorm_min(),
                std::numeric_limits<Scalar>::lowest(),
            },
    };
    const auto mixed_radiance = SpectrumFor<Scalar>{
        .values =
            {
                std::numeric_limits<Scalar>::max(),
                Scalar{-4.5},
                Scalar{0},
                std::numeric_limits<Scalar>::denorm_min(),
            },
    };
    const auto mixed_wavelengths =
        make_wavelengths<Scalar>({Scalar{360}, Scalar{830}, Scalar{361}, Scalar{829}},
                                 {Scalar{4}, Scalar{0.5}, std::numeric_limits<Scalar>::denorm_min(),
                                  std::numeric_limits<Scalar>::max()});
    constexpr auto mixed_flags =
        PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces;
    const auto mixed =
        StateFor<Scalar>::create(mixed_beta, mixed_radiance,
                                 PathDepthCounters{
                                     .diffuse = 1,
                                     .glossy = 1,
                                     .specular = 1,
                                     .transmission = 1,
                                     .volume = 1,
                                 },
                                 std::numeric_limits<Scalar>::max(), mixed_wavelengths, mixed_flags,
                                 MediumId{.value = std::numeric_limits<std::uint32_t>::max()});
    if (!mixed.has_value()) {
        return std::unexpected(mixed.error());
    }

    return std::vector<StateFor<Scalar>>{*primary, *first, *mixed};
}

template <SpectrumScalar Scalar>
void expect_states_bitwise_equal(const StateFor<Scalar>& actual, const StateFor<Scalar>& expected) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expect_scalar_bits_equal(actual.beta()[lane], expected.beta()[lane]);
        expect_scalar_bits_equal(actual.accumulated_radiance()[lane],
                                 expected.accumulated_radiance()[lane]);
        expect_scalar_bits_equal(actual.wavelengths()[lane].nanometers,
                                 expected.wavelengths()[lane].nanometers);
        expect_scalar_bits_equal(actual.wavelengths()[lane].probability.value,
                                 expected.wavelengths()[lane].probability.value);
        EXPECT_EQ(actual.wavelengths()[lane].probability.measure,
                  expected.wavelengths()[lane].probability.measure);
    }
    EXPECT_EQ(actual.depth(), expected.depth());
    EXPECT_EQ(actual.depth_counters(), expected.depth_counters());
    expect_scalar_bits_equal(actual.eta_scale(), expected.eta_scale());
    EXPECT_EQ(actual.delta_flags(), expected.delta_flags());
    EXPECT_EQ(actual.current_medium(), expected.current_medium());
}

template <SpectrumScalar Scalar>
void expect_column_layout(const SoAFor<Scalar>& soa,
                          const std::span<const StateFor<Scalar>> states) {
    const auto& columns = soa.columns();
    EXPECT_EQ(columns.schema_version, CurrentPathStateSoASchemaVersion);
    EXPECT_EQ(columns.path_count, states.size());

    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        ASSERT_EQ(columns.beta[lane].size(), states.size());
        ASSERT_EQ(columns.accumulated_radiance[lane].size(), states.size());
        ASSERT_EQ(columns.wavelength_nanometers[lane].size(), states.size());
        ASSERT_EQ(columns.wavelength_pdf_values[lane].size(), states.size());
        ASSERT_EQ(columns.wavelength_pdf_measures[lane].size(), states.size());
    }
    ASSERT_EQ(columns.depth.size(), states.size());
    ASSERT_EQ(columns.depth_counters.diffuse.size(), states.size());
    ASSERT_EQ(columns.depth_counters.glossy.size(), states.size());
    ASSERT_EQ(columns.depth_counters.specular.size(), states.size());
    ASSERT_EQ(columns.depth_counters.transmission.size(), states.size());
    ASSERT_EQ(columns.depth_counters.volume.size(), states.size());
    ASSERT_EQ(columns.eta_scale.size(), states.size());
    ASSERT_EQ(columns.delta_flags.size(), states.size());
    ASSERT_EQ(columns.current_medium_values.size(), states.size());

    for (auto index = std::size_t{}; index < states.size(); ++index) {
        const auto& expected = states[index];
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            expect_scalar_bits_equal(columns.beta[lane][index], expected.beta()[lane]);
            expect_scalar_bits_equal(columns.accumulated_radiance[lane][index],
                                     expected.accumulated_radiance()[lane]);
            expect_scalar_bits_equal(columns.wavelength_nanometers[lane][index],
                                     expected.wavelengths()[lane].nanometers);
            expect_scalar_bits_equal(columns.wavelength_pdf_values[lane][index],
                                     expected.wavelengths()[lane].probability.value);
            EXPECT_EQ(columns.wavelength_pdf_measures[lane][index],
                      expected.wavelengths()[lane].probability.measure);
        }

        EXPECT_EQ(columns.depth[index], expected.depth());
        EXPECT_EQ(columns.depth_counters.diffuse[index], expected.depth_counters().diffuse);
        EXPECT_EQ(columns.depth_counters.glossy[index], expected.depth_counters().glossy);
        EXPECT_EQ(columns.depth_counters.specular[index], expected.depth_counters().specular);
        EXPECT_EQ(columns.depth_counters.transmission[index],
                  expected.depth_counters().transmission);
        EXPECT_EQ(columns.depth_counters.volume[index], expected.depth_counters().volume);
        expect_scalar_bits_equal(columns.eta_scale[index], expected.eta_scale());
        EXPECT_EQ(columns.delta_flags[index], expected.delta_flags());
        EXPECT_EQ(columns.current_medium_values[index], expected.current_medium().value);
    }
}

template <SpectrumScalar Scalar> void expect_exact_mapping_and_ownership() {
    const auto expected_result = make_states<Scalar>();
    ASSERT_TRUE(expected_result.has_value());
    const auto expected = *expected_result;
    auto source = expected;

    auto soa_result = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{source},
                                               CurrentPathStateSoASchemaVersion);
    ASSERT_TRUE(soa_result.has_value());
    auto soa = std::move(*soa_result);

    source.assign(source.size(), source.front());
    EXPECT_EQ(soa.schema_version(), CurrentPathStateSoASchemaVersion);
    EXPECT_EQ(soa.size(), expected.size());
    EXPECT_FALSE(soa.empty());
    expect_column_layout<Scalar>(soa, std::span<const StateFor<Scalar>>{expected});

    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        const auto state = soa.at(index);
        ASSERT_TRUE(state.has_value());
        expect_states_bitwise_equal<Scalar>(*state, expected[index]);
    }

    const auto round_trip = soa.to_aos();
    ASSERT_TRUE(round_trip.has_value());
    ASSERT_EQ(round_trip->size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        expect_states_bitwise_equal<Scalar>((*round_trip)[index], expected[index]);
    }

    const auto replay = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{*round_trip},
                                                 CurrentPathStateSoASchemaVersion);
    ASSERT_TRUE(replay.has_value());
    const auto replay_aos = replay->to_aos();
    ASSERT_TRUE(replay_aos.has_value());
    ASSERT_EQ(replay_aos->size(), round_trip->size());
    for (auto index = std::size_t{}; index < round_trip->size(); ++index) {
        expect_states_bitwise_equal<Scalar>((*replay_aos)[index], (*round_trip)[index]);
    }
}

TEST(PathStateSoATest, MapsEveryTransportAndReferenceColumnBitForBit) {
    static_assert(CurrentPathStateSoASchemaVersion == 1U);
    static_assert(std::same_as<PathStateSoA, PathStateSoAT<TransportScalar>>);
    static_assert(std::same_as<ReferencePathStateSoA, PathStateSoAT<ReferenceScalar>>);
    static_assert(!std::same_as<PathStateSoA, ReferencePathStateSoA>);
    static_assert(!std::convertible_to<PathStateSoA, ReferencePathStateSoA>);
    static_assert(!std::convertible_to<ReferencePathStateSoA, PathStateSoA>);
    static_assert(!HasRvalueColumns<PathStateSoA>);
    static_assert(!HasRvalueColumns<const PathStateSoA>);
    static_assert(!HasRvalueColumns<ReferencePathStateSoA>);
    static_assert(!HasRvalueColumns<const ReferencePathStateSoA>);

    expect_exact_mapping_and_ownership<TransportScalar>();
    expect_exact_mapping_and_ownership<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_empty_mapping() {
    const auto soa = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{},
                                              CurrentPathStateSoASchemaVersion);
    ASSERT_TRUE(soa.has_value());
    EXPECT_EQ(soa->schema_version(), CurrentPathStateSoASchemaVersion);
    EXPECT_EQ(soa->size(), 0U);
    EXPECT_TRUE(soa->empty());

    expect_column_layout<Scalar>(*soa, std::span<const StateFor<Scalar>>{});

    const auto exported = soa->to_aos();
    ASSERT_TRUE(exported.has_value());
    EXPECT_TRUE(exported->empty());

    const auto missing = soa->at(0);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::invalid_argument);
}

TEST(PathStateSoATest, RepresentsAnEmptyVersionedMappingWithoutSentinelPaths) {
    expect_empty_mapping<TransportScalar>();
    expect_empty_mapping<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_unsupported_versions_rejected() {
    const auto states = make_states<Scalar>();
    ASSERT_TRUE(states.has_value());

    for (const auto version : std::array{std::uint32_t{0}, std::uint32_t{2},
                                         std::numeric_limits<std::uint32_t>::max()}) {
        const auto populated =
            SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{*states}, version);
        ASSERT_FALSE(populated.has_value());
        EXPECT_EQ(populated.error().code, core::StatusCode::incompatible);
        EXPECT_EQ(populated.error().message, "Unsupported path state SoA schema version " +
                                                 std::to_string(version) + "; expected 1.");

        const auto empty = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{}, version);
        ASSERT_FALSE(empty.has_value());
        EXPECT_EQ(empty.error().code, core::StatusCode::incompatible);
        EXPECT_EQ(empty.error().message, populated.error().message);
    }
}

TEST(PathStateSoATest, RejectsUnsupportedVersionsBeforeMappingWithoutFallback) {
    expect_unsupported_versions_rejected<TransportScalar>();
    expect_unsupported_versions_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_bounds_rejected_without_mutation() {
    const auto states = make_states<Scalar>();
    ASSERT_TRUE(states.has_value());
    const auto soa = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{*states},
                                              CurrentPathStateSoASchemaVersion);
    ASSERT_TRUE(soa.has_value());

    const auto before = soa->to_aos();
    ASSERT_TRUE(before.has_value());
    for (const auto index : std::array{soa->size(), std::numeric_limits<std::size_t>::max()}) {
        const auto missing = soa->at(index);
        ASSERT_FALSE(missing.has_value());
        EXPECT_EQ(missing.error().code, core::StatusCode::invalid_argument);
    }

    const auto after = soa->to_aos();
    ASSERT_TRUE(after.has_value());
    ASSERT_EQ(after->size(), before->size());
    for (auto index = std::size_t{}; index < before->size(); ++index) {
        expect_states_bitwise_equal<Scalar>((*after)[index], (*before)[index]);
    }
}

TEST(PathStateSoATest, RejectsOutOfBoundsReadsWithoutReturningOrMutatingAPath) {
    expect_bounds_rejected_without_mutation<TransportScalar>();
    expect_bounds_rejected_without_mutation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_move_preserves_invariants() {
    const auto states = make_states<Scalar>();
    ASSERT_TRUE(states.has_value());
    auto source_result = SoAFor<Scalar>::from_aos(std::span<const StateFor<Scalar>>{*states},
                                                  CurrentPathStateSoASchemaVersion);
    ASSERT_TRUE(source_result.has_value());
    auto source = std::move(*source_result);
    auto destination = std::move(source);

    EXPECT_EQ(source.schema_version(), CurrentPathStateSoASchemaVersion);
    EXPECT_EQ(source.size(), 0U);
    EXPECT_TRUE(source.empty());
    expect_column_layout<Scalar>(source, std::span<const StateFor<Scalar>>{});
    const auto moved_from_aos = source.to_aos();
    ASSERT_TRUE(moved_from_aos.has_value());
    EXPECT_TRUE(moved_from_aos->empty());

    EXPECT_EQ(destination.schema_version(), CurrentPathStateSoASchemaVersion);
    EXPECT_EQ(destination.size(), states->size());
    EXPECT_FALSE(destination.empty());
    expect_column_layout<Scalar>(destination, std::span<const StateFor<Scalar>>{*states});
    const auto destination_aos = destination.to_aos();
    ASSERT_TRUE(destination_aos.has_value());
    ASSERT_EQ(destination_aos->size(), states->size());
    for (auto index = std::size_t{}; index < states->size(); ++index) {
        expect_states_bitwise_equal<Scalar>((*destination_aos)[index], (*states)[index]);
    }
}

TEST(PathStateSoATest, MoveConstructionPreservesDestinationAndEmptiesSource) {
    expect_move_preserves_invariants<TransportScalar>();
    expect_move_preserves_invariants<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
