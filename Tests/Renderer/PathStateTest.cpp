#include <Blackframe/Renderer/PathStateDiagnostics.hpp>
#include <Blackframe/Renderer/TransportInterfaces.hpp>
#include <array>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using WavelengthsFor = SampledWavelengthsT<Scalar>;

template <SpectrumScalar Scalar>
using StateFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, PathState, ReferencePathState>;

inline constexpr std::string_view BetaError =
    "Path beta requires every spectral lane to be finite.";
inline constexpr std::string_view RadianceError =
    "Path accumulated radiance requires every spectral lane to be finite.";
inline constexpr std::string_view EtaScaleError =
    "Path eta scale must be finite and strictly positive.";
inline constexpr std::string_view WavelengthError =
    "Path wavelengths must be finite values in [360, 830] nanometers.";
inline constexpr std::string_view WavelengthPdfError =
    "Path wavelength PDFs must be finite, strictly positive, and use wavelength measure.";
inline constexpr std::string_view DeltaBitsError = "Path delta flags contain unsupported bits.";
inline constexpr std::string_view PrimaryDeltaError =
    "A primary path cannot report completed-bounce delta flags.";
inline constexpr std::string_view OneBounceDeltaError =
    "A one-bounce path cannot combine a delta previous bounce with non-delta history.";
inline constexpr std::string_view DeltaHistoryError =
    "A non-delta previous bounce must be recorded in the path history.";
inline constexpr std::string_view CounterHistoryError =
    "Path depth counters are inconsistent with the delta history.";

[[nodiscard]] constexpr PathDepthCounters counters_for(const std::uint32_t depth,
                                                       const PathDeltaFlags flags) noexcept {
    if (depth == 0) {
        return {};
    }
    if (has_path_delta_flag(flags, PathDeltaFlags::previous_bounce_was_delta) &&
        has_path_delta_flag(flags, PathDeltaFlags::any_non_delta_bounces) && depth > 1) {
        return PathDepthCounters{
            .diffuse = depth - 1U,
            .specular = 1,
        };
    }
    if (has_path_delta_flag(flags, PathDeltaFlags::previous_bounce_was_delta)) {
        return PathDepthCounters{.specular = depth};
    }
    return PathDepthCounters{.diffuse = depth};
}

template <typename Result>
void expect_invalid(const Result& result, const std::string_view expected_message) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, expected_message);
}

template <SpectrumScalar Scalar> [[nodiscard]] WavelengthsFor<Scalar> standard_wavelengths() {
    const auto nanometers = std::array{Scalar{360}, Scalar{470}, Scalar{600}, Scalar{830}};
    const auto probabilities = std::array{Scalar{0.5}, Scalar{0.25}, Scalar{0.125}, Scalar{0.0625}};
    auto result = WavelengthsFor<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = typename WavelengthsFor<Scalar>::value_type{
            .nanometers = nanometers[lane],
            .probability =
                {
                    .value = probabilities[lane],
                    .measure = ProbabilityMeasure::wavelength,
                },
        };
    }
    return result;
}

template <SpectrumScalar Scalar> [[nodiscard]] WavelengthsFor<Scalar> boundary_wavelengths() {
    const auto nanometers = std::array{Scalar{830}, Scalar{360}, Scalar{360}, Scalar{830}};
    const auto probabilities = std::array{std::numeric_limits<Scalar>::denorm_min(), Scalar{0.25},
                                          Scalar{2}, std::numeric_limits<Scalar>::max()};
    auto result = WavelengthsFor<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = typename WavelengthsFor<Scalar>::value_type{
            .nanometers = nanometers[lane],
            .probability =
                {
                    .value = probabilities[lane],
                    .measure = ProbabilityMeasure::wavelength,
                },
        };
    }
    return result;
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> signed_beta() {
    return {
        .values =
            {
                std::numeric_limits<Scalar>::lowest(),
                -std::numeric_limits<Scalar>::denorm_min(),
                -Scalar{0},
                std::numeric_limits<Scalar>::max(),
            },
    };
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> signed_radiance() {
    return {
        .values =
            {
                std::numeric_limits<Scalar>::max(),
                Scalar{-1},
                Scalar{0},
                std::numeric_limits<Scalar>::denorm_min(),
            },
    };
}

template <SpectrumScalar Scalar> void expect_exact_initial_state() {
    auto wavelengths = standard_wavelengths<Scalar>();
    const auto expected_wavelengths = wavelengths;
    constexpr auto medium = MediumId{.value = std::numeric_limits<std::uint32_t>::max()};
    const auto state = StateFor<Scalar>::create_initial(wavelengths, medium);
    ASSERT_TRUE(state.has_value());

    auto expected_beta = SpectrumFor<Scalar>{};
    expected_beta.values.fill(Scalar{1});
    EXPECT_EQ(state->beta(), expected_beta);
    EXPECT_EQ(state->accumulated_radiance(), SpectrumFor<Scalar>{});
    EXPECT_EQ(state->depth(), 0U);
    EXPECT_EQ(state->depth_counters(), PathDepthCounters{});
    EXPECT_EQ(state->eta_scale(), Scalar{1});
    EXPECT_EQ(state->wavelengths(), expected_wavelengths);
    EXPECT_EQ(state->delta_flags(), PathDeltaFlags::none);
    EXPECT_EQ(state->current_medium(), medium);

    wavelengths[0].nanometers = Scalar{400};
    wavelengths[0].probability.value = Scalar{4};
    EXPECT_EQ(state->wavelengths(), expected_wavelengths);
}

TEST(PathStateTest, CreatesAnExactPrimaryWithoutWavelengthPreweighting) {
    static_assert(std::same_as<decltype(PathState::create_initial(
                                   std::declval<SampledWavelengths>(), VacuumMedium)),
                               core::Result<PathState>>);
    static_assert(std::same_as<decltype(ReferencePathState::create_initial(
                                   std::declval<ReferenceSampledWavelengths>(), VacuumMedium)),
                               core::Result<ReferencePathState>>);
    static_assert(!std::same_as<PathState, ReferencePathState>);
    static_assert(!std::default_initializable<PathState>);
    static_assert(!std::default_initializable<ReferencePathState>);
    static_assert(std::is_standard_layout_v<PathState>);
    static_assert(std::is_trivially_copyable_v<PathState>);
    static_assert(std::is_standard_layout_v<ReferencePathState>);
    static_assert(std::is_trivially_copyable_v<ReferencePathState>);

    expect_exact_initial_state<TransportScalar>();
    expect_exact_initial_state<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_explicit_state_boundaries() {
    const auto beta = signed_beta<Scalar>();
    const auto radiance = signed_radiance<Scalar>();
    const auto wavelengths = boundary_wavelengths<Scalar>();
    constexpr auto combined_flags =
        PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces;
    constexpr auto medium = MediumId{.value = std::numeric_limits<std::uint32_t>::max()};
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    constexpr auto maximum_counters = PathDepthCounters{
        .diffuse = maximum - 1U,
        .specular = 1,
        .transmission = 1,
    };

    for (const auto eta_scale : std::array{std::numeric_limits<Scalar>::denorm_min(), Scalar{1},
                                           std::numeric_limits<Scalar>::max()}) {
        const auto state = StateFor<Scalar>::create(beta, radiance, maximum_counters, eta_scale,
                                                    wavelengths, combined_flags, medium);
        ASSERT_TRUE(state.has_value());
        EXPECT_EQ(state->beta(), beta);
        EXPECT_EQ(state->accumulated_radiance(), radiance);
        EXPECT_EQ(state->depth(), maximum);
        EXPECT_EQ(state->depth_counters(), maximum_counters);
        EXPECT_EQ(state->eta_scale(), eta_scale);
        EXPECT_EQ(state->wavelengths(), wavelengths);
        EXPECT_EQ(state->delta_flags(), combined_flags);
        EXPECT_EQ(state->current_medium(), medium);
    }

    for (const auto [depth, flags] : std::array{
             std::pair{0U, PathDeltaFlags::none},
             std::pair{1U, PathDeltaFlags::previous_bounce_was_delta},
             std::pair{1U, PathDeltaFlags::any_non_delta_bounces}, std::pair{2U, combined_flags}}) {
        const auto depth_counters = counters_for(depth, flags);
        const auto state = StateFor<Scalar>::create(beta, radiance, depth_counters, Scalar{1},
                                                    wavelengths, flags, medium);
        ASSERT_TRUE(state.has_value());
        EXPECT_EQ(state->depth(), depth);
        EXPECT_EQ(state->depth_counters(), depth_counters);
        EXPECT_EQ(state->delta_flags(), flags);
    }

    EXPECT_TRUE(has_path_delta_flag(combined_flags, PathDeltaFlags::previous_bounce_was_delta));
    EXPECT_TRUE(has_path_delta_flag(combined_flags, PathDeltaFlags::any_non_delta_bounces));
    EXPECT_FALSE(
        has_path_delta_flag(PathDeltaFlags::none, PathDeltaFlags::previous_bounce_was_delta));
    EXPECT_TRUE(has_path_delta_flag(PathDeltaFlags::none, PathDeltaFlags::none));
    EXPECT_FALSE(has_path_delta_flag(combined_flags, PathDeltaFlags::none));
}

TEST(PathStateTest, StoresSignedSpectraNumericLimitsDeltaHistoryAndOpaqueMediumExactly) {
    expect_explicit_state_boundaries<TransportScalar>();
    expect_explicit_state_boundaries<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_non_finite_spectra_rejected() {
    const auto wavelengths = standard_wavelengths<Scalar>();
    const auto valid_beta = signed_beta<Scalar>();
    const auto valid_radiance = signed_radiance<Scalar>();
    const auto infinity = std::numeric_limits<Scalar>::infinity();

    for (const auto invalid :
         std::array{std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto beta = valid_beta;
            beta[lane] = invalid;
            expect_invalid(StateFor<Scalar>::create(beta, valid_radiance, {}, Scalar{1},
                                                    wavelengths, PathDeltaFlags::none,
                                                    VacuumMedium),
                           BetaError);

            auto radiance = valid_radiance;
            radiance[lane] = invalid;
            expect_invalid(StateFor<Scalar>::create(valid_beta, radiance, {}, Scalar{1},
                                                    wavelengths, PathDeltaFlags::none,
                                                    VacuumMedium),
                           RadianceError);
        }
    }
}

TEST(PathStateTest, RejectsEveryNonFiniteSpectralLaneWithoutRepair) {
    expect_non_finite_spectra_rejected<TransportScalar>();
    expect_non_finite_spectra_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_eta_and_wavelengths_rejected() {
    const auto beta = signed_beta<Scalar>();
    const auto radiance = signed_radiance<Scalar>();
    const auto valid_wavelengths = standard_wavelengths<Scalar>();
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto create = [&](const Scalar eta_scale, const WavelengthsFor<Scalar>& wavelengths) {
        return StateFor<Scalar>::create(beta, radiance, {}, eta_scale, wavelengths,
                                        PathDeltaFlags::none, VacuumMedium);
    };

    for (const auto invalid_eta :
         std::array{Scalar{0}, -Scalar{0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(create(invalid_eta, valid_wavelengths), EtaScaleError);
    }

    expect_invalid(create(Scalar{1}, WavelengthsFor<Scalar>{}), WavelengthError);

    const auto below_visible =
        std::nextafter(Scalar{VisibleWavelengthMinimumNanometers}, -infinity);
    const auto above_visible = std::nextafter(Scalar{VisibleWavelengthMaximumNanometers}, infinity);
    for (const auto invalid_wavelength :
         std::array{below_visible, above_visible, std::numeric_limits<Scalar>::quiet_NaN(),
                    infinity, -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto wavelengths = valid_wavelengths;
            wavelengths[lane].nanometers = invalid_wavelength;
            expect_invalid(create(Scalar{1}, wavelengths), WavelengthError);
        }
    }

    for (const auto invalid_pdf :
         std::array{Scalar{0}, -Scalar{0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto wavelengths = valid_wavelengths;
            wavelengths[lane].probability.value = invalid_pdf;
            expect_invalid(create(Scalar{1}, wavelengths), WavelengthPdfError);
        }
    }

    for (const auto invalid_measure :
         std::array{ProbabilityMeasure::discrete, ProbabilityMeasure::solid_angle,
                    ProbabilityMeasure::area, ProbabilityMeasure::distance,
                    ProbabilityMeasure::volume, static_cast<ProbabilityMeasure>(255)}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto wavelengths = valid_wavelengths;
            wavelengths[lane].probability.measure = invalid_measure;
            expect_invalid(create(Scalar{1}, wavelengths), WavelengthPdfError);
        }
    }
}

TEST(PathStateTest, RejectsInvalidEtaWavelengthsAndPdfsWithoutSubstitution) {
    expect_invalid_eta_and_wavelengths_rejected<TransportScalar>();
    expect_invalid_eta_and_wavelengths_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_incoherent_delta_flags_rejected() {
    const auto beta = signed_beta<Scalar>();
    const auto radiance = signed_radiance<Scalar>();
    const auto wavelengths = standard_wavelengths<Scalar>();
    const auto create = [&](const std::uint32_t depth, const PathDeltaFlags flags) {
        return StateFor<Scalar>::create(beta, radiance, counters_for(depth, flags), Scalar{1},
                                        wavelengths, flags, VacuumMedium);
    };

    expect_invalid(create(1, static_cast<PathDeltaFlags>(0x80U)), DeltaBitsError);
    for (const auto invalid_primary : std::array{
             PathDeltaFlags::previous_bounce_was_delta, PathDeltaFlags::any_non_delta_bounces,
             PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces}) {
        expect_invalid(create(0, invalid_primary), PrimaryDeltaError);
    }
    expect_invalid(create(1, PathDeltaFlags::previous_bounce_was_delta |
                                 PathDeltaFlags::any_non_delta_bounces),
                   OneBounceDeltaError);
    expect_invalid(create(1, PathDeltaFlags::none), DeltaHistoryError);
}

TEST(PathStateTest, RejectsUnknownAndIncoherentDeltaFlags) {
    expect_incoherent_delta_flags_rejected<TransportScalar>();
    expect_incoherent_delta_flags_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_incoherent_depth_counters_rejected() {
    const auto beta = signed_beta<Scalar>();
    const auto radiance = signed_radiance<Scalar>();
    const auto wavelengths = standard_wavelengths<Scalar>();
    const auto create = [&](const PathDepthCounters counters, const PathDeltaFlags flags) {
        return StateFor<Scalar>::create(beta, radiance, counters, Scalar{1}, wavelengths, flags,
                                        VacuumMedium);
    };

    expect_invalid(
        create(PathDepthCounters{.diffuse = 1}, PathDeltaFlags::previous_bounce_was_delta),
        CounterHistoryError);
    expect_invalid(create(PathDepthCounters{.specular = 1}, PathDeltaFlags::any_non_delta_bounces),
                   CounterHistoryError);
    expect_invalid(
        create(PathDepthCounters{.diffuse = 2},
               PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces),
        CounterHistoryError);
    expect_invalid(
        create(PathDepthCounters{.specular = 2},
               PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces),
        CounterHistoryError);

    const auto volume =
        create(PathDepthCounters{.volume = 1}, PathDeltaFlags::any_non_delta_bounces);
    ASSERT_TRUE(volume.has_value());
    EXPECT_EQ(volume->depth(), 1U);
    EXPECT_EQ(volume->depth_counters(), (PathDepthCounters{.volume = 1}));
    expect_invalid(
        create(PathDepthCounters{.volume = 1}, PathDeltaFlags::previous_bounce_was_delta),
        CounterHistoryError);

    const auto impossible_transmission = create(PathDepthCounters{.diffuse = 1, .transmission = 2},
                                                PathDeltaFlags::any_non_delta_bounces);
    ASSERT_FALSE(impossible_transmission.has_value());
    EXPECT_EQ(impossible_transmission.error().code, core::StatusCode::invalid_argument);

    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    const auto overflowing = create(PathDepthCounters{.diffuse = maximum, .glossy = 1},
                                    PathDeltaFlags::any_non_delta_bounces);
    ASSERT_FALSE(overflowing.has_value());
    EXPECT_EQ(overflowing.error().code, core::StatusCode::resource_exhausted);
}

TEST(PathStateTest, BindsExactValidatedDepthCountersToThePath) {
    expect_incoherent_depth_counters_rejected<TransportScalar>();
    expect_incoherent_depth_counters_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<StateFor<Scalar>> make_diagnostic_state() {
    const auto beta = SpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{0.5}, Scalar{2}, -Scalar{0}},
    };
    const auto radiance = SpectrumFor<Scalar>{
        .values = {Scalar{0.25}, Scalar{4}, Scalar{8}, Scalar{16}},
    };
    constexpr auto flags =
        PathDeltaFlags::previous_bounce_was_delta | PathDeltaFlags::any_non_delta_bounces;
    constexpr auto counters = PathDepthCounters{
        .diffuse = 1,
        .glossy = 1,
        .specular = 1,
        .transmission = 1,
        .volume = 1,
    };
    return StateFor<Scalar>::create(beta, radiance, counters, Scalar{1.5},
                                    standard_wavelengths<Scalar>(), flags, MediumId{.value = 42});
}

TEST(PathStateDiagnosticTest, DumpsTransportAndReferencePathsWithStableExactBits) {
    const auto transport = make_diagnostic_state<TransportScalar>();
    const auto reference = make_diagnostic_state<ReferenceScalar>();
    ASSERT_TRUE(transport.has_value());
    ASSERT_TRUE(reference.has_value());

    constexpr auto expected_transport =
        R"({"schema_version":2,"precision":"float32","throughput_bits":["0x3f800000","0x3f000000","0x40000000","0x80000000"],"accumulated_radiance_bits":["0x3e800000","0x40800000","0x41000000","0x41800000"],"depth":4,"depth_counters":{"diffuse":1,"glossy":1,"specular":1,"transmission":1,"volume":1},"eta_scale_bits":"0x3fc00000","wavelengths":[{"nanometers_bits":"0x43b40000","pdf_bits":"0x3f000000","measure":"wavelength"},{"nanometers_bits":"0x43eb0000","pdf_bits":"0x3e800000","measure":"wavelength"},{"nanometers_bits":"0x44160000","pdf_bits":"0x3e000000","measure":"wavelength"},{"nanometers_bits":"0x444f8000","pdf_bits":"0x3d800000","measure":"wavelength"}],"delta_flags":"0x03","current_medium":"0x0000002a"})";
    constexpr auto expected_reference =
        R"({"schema_version":2,"precision":"float64","throughput_bits":["0x3ff0000000000000","0x3fe0000000000000","0x4000000000000000","0x8000000000000000"],"accumulated_radiance_bits":["0x3fd0000000000000","0x4010000000000000","0x4020000000000000","0x4030000000000000"],"depth":4,"depth_counters":{"diffuse":1,"glossy":1,"specular":1,"transmission":1,"volume":1},"eta_scale_bits":"0x3ff8000000000000","wavelengths":[{"nanometers_bits":"0x4076800000000000","pdf_bits":"0x3fe0000000000000","measure":"wavelength"},{"nanometers_bits":"0x407d600000000000","pdf_bits":"0x3fd0000000000000","measure":"wavelength"},{"nanometers_bits":"0x4082c00000000000","pdf_bits":"0x3fc0000000000000","measure":"wavelength"},{"nanometers_bits":"0x4089f00000000000","pdf_bits":"0x3fb0000000000000","measure":"wavelength"}],"delta_flags":"0x03","current_medium":"0x0000002a"})";

    const auto transport_dump =
        serialize_path_state_diagnostic(*transport, CurrentPathStateDiagnosticSchemaVersion);
    const auto reference_dump =
        serialize_path_state_diagnostic(*reference, CurrentPathStateDiagnosticSchemaVersion);
    ASSERT_TRUE(transport_dump.has_value());
    ASSERT_TRUE(reference_dump.has_value());
    EXPECT_EQ(*transport_dump, expected_transport);
    EXPECT_EQ(*reference_dump, expected_reference);
    const auto replay =
        serialize_path_state_diagnostic(*transport, CurrentPathStateDiagnosticSchemaVersion);
    ASSERT_TRUE(replay.has_value());
    EXPECT_EQ(*replay, *transport_dump);
}

TEST(PathStateDiagnosticTest, RejectsUnsupportedSchemaVersionsWithoutFallback) {
    const auto state = make_diagnostic_state<TransportScalar>();
    ASSERT_TRUE(state.has_value());

    for (const auto version : std::array{0U, 1U, 3U}) {
        const auto dump = serialize_path_state_diagnostic(*state, version);
        ASSERT_FALSE(dump.has_value());
        EXPECT_EQ(dump.error().code, core::StatusCode::incompatible);
        EXPECT_EQ(dump.error().message, "Unsupported path state diagnostic schema version " +
                                            std::to_string(version) + "; expected 2.");
    }
}

} // namespace
} // namespace blackframe::renderer
