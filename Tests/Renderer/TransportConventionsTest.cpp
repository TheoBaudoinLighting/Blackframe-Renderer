#include <Blackframe/Renderer/TransportConventionDiagnostics.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(TransportConventionsTest, FreezesProbabilityMeasuresAndTransportModes) {
    static_assert(std::is_same_v<std::underlying_type_t<ProbabilityMeasure>, std::uint8_t>);
    static_assert(std::is_same_v<std::underlying_type_t<TransportMode>, std::uint8_t>);
    static_assert(sizeof(ProbabilityMeasure) == 1U);
    static_assert(sizeof(TransportMode) == 1U);
    static_assert(std::is_standard_layout_v<ProbabilityDensity>);
    static_assert(std::is_trivially_copyable_v<ProbabilityDensity>);
    static_assert(std::is_standard_layout_v<ReferenceProbabilityDensity>);
    static_assert(std::is_trivially_copyable_v<ReferenceProbabilityDensity>);

    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::discrete), 0U);
    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::solid_angle), 1U);
    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::area), 2U);
    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::distance), 3U);
    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::volume), 4U);
    EXPECT_EQ(static_cast<std::uint8_t>(ProbabilityMeasure::wavelength), 5U);
    for (const auto measure : std::array{
             ProbabilityMeasure::discrete,
             ProbabilityMeasure::solid_angle,
             ProbabilityMeasure::area,
             ProbabilityMeasure::distance,
             ProbabilityMeasure::volume,
             ProbabilityMeasure::wavelength,
         }) {
        EXPECT_TRUE(is_known_probability_measure(measure));
    }
    EXPECT_FALSE(is_known_probability_measure(static_cast<ProbabilityMeasure>(6U)));
    EXPECT_FALSE(is_known_probability_measure(static_cast<ProbabilityMeasure>(0xffU)));
    EXPECT_EQ(ContinuousBsdfProbabilityMeasure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ(DeltaBsdfProbabilityMeasure, ProbabilityMeasure::discrete);

    EXPECT_EQ(static_cast<std::uint8_t>(TransportMode::radiance), 0U);
    EXPECT_EQ(static_cast<std::uint8_t>(TransportMode::importance), 1U);
    EXPECT_TRUE(is_known_transport_mode(TransportMode::radiance));
    EXPECT_TRUE(is_known_transport_mode(TransportMode::importance));
    EXPECT_FALSE(is_known_transport_mode(static_cast<TransportMode>(2U)));
}

TEST(TransportConventionsTest, FreezesLobeBitsMasksAndConcreteEventRules) {
    static_assert(std::is_same_v<std::underlying_type_t<ScatteringLobe>, std::uint32_t>);
    static_assert(sizeof(ScatteringLobe) == 4U);

    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::none), 0x00000000U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::diffuse), 0x00000001U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::glossy), 0x00000002U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::specular), 0x00000004U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::reflection), 0x00000008U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::transmission), 0x00000010U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringLobe::volume), 0x00000020U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringFamilyMask), 0x00000007U);
    EXPECT_EQ(scattering_lobe_bits(ScatteringDirectionMask), 0x00000018U);
    EXPECT_EQ(scattering_lobe_bits(KnownScatteringLobeMask), 0x0000003fU);

    constexpr auto diffuse_reflection = ScatteringLobe::diffuse | ScatteringLobe::reflection;
    EXPECT_TRUE(has_scattering_lobe(diffuse_reflection, ScatteringLobe::diffuse));
    EXPECT_TRUE(has_scattering_lobe(diffuse_reflection, ScatteringLobe::reflection));
    EXPECT_FALSE(has_scattering_lobe(diffuse_reflection, ScatteringLobe::glossy));
    EXPECT_FALSE(has_scattering_lobe(diffuse_reflection, ScatteringLobe::none));
    EXPECT_TRUE(has_scattering_lobe(ScatteringLobe::none, ScatteringLobe::none));
    EXPECT_TRUE(is_known_scattering_lobe_mask(ScatteringLobe::none));
    EXPECT_TRUE(is_known_scattering_lobe_mask(KnownScatteringLobeMask));
    EXPECT_FALSE(is_known_scattering_lobe_mask(static_cast<ScatteringLobe>(0x80000000U)));

    constexpr auto valid_surface_events = std::array{
        ScatteringLobe::diffuse | ScatteringLobe::reflection,
        ScatteringLobe::diffuse | ScatteringLobe::transmission,
        ScatteringLobe::glossy | ScatteringLobe::reflection,
        ScatteringLobe::glossy | ScatteringLobe::transmission,
        ScatteringLobe::specular | ScatteringLobe::reflection,
        ScatteringLobe::specular | ScatteringLobe::transmission,
    };
    for (const auto event : valid_surface_events) {
        EXPECT_TRUE(is_valid_surface_scattering_event(event));
        EXPECT_TRUE(is_valid_scattering_event(event));
    }
    EXPECT_TRUE(is_valid_scattering_event(ScatteringLobe::volume));
    EXPECT_FALSE(is_valid_surface_scattering_event(ScatteringLobe::volume));

    constexpr auto invalid_events = std::array{
        ScatteringLobe::none,
        ScatteringLobe::diffuse,
        ScatteringLobe::reflection,
        ScatteringLobe::diffuse | ScatteringLobe::glossy | ScatteringLobe::reflection,
        ScatteringLobe::diffuse | ScatteringLobe::reflection | ScatteringLobe::transmission,
        ScatteringLobe::volume | ScatteringLobe::reflection,
        static_cast<ScatteringLobe>(0x80000000U),
    };
    for (const auto event : invalid_events) {
        EXPECT_FALSE(is_valid_scattering_event(event));
    }
}

TEST(TransportConventionsTest, ClassifiesSpecularAsDeltaWithoutReclassifyingMasksOrVolumes) {
    constexpr auto specular_reflection = ScatteringLobe::specular | ScatteringLobe::reflection;
    constexpr auto specular_transmission = ScatteringLobe::specular | ScatteringLobe::transmission;
    constexpr auto diffuse_reflection = ScatteringLobe::diffuse | ScatteringLobe::reflection;
    constexpr auto glossy_transmission = ScatteringLobe::glossy | ScatteringLobe::transmission;

    EXPECT_TRUE(is_delta_surface_scattering_event(specular_reflection));
    EXPECT_TRUE(is_delta_surface_scattering_event(specular_transmission));
    EXPECT_FALSE(is_continuous_surface_scattering_event(specular_reflection));
    EXPECT_TRUE(is_continuous_surface_scattering_event(diffuse_reflection));
    EXPECT_TRUE(is_continuous_surface_scattering_event(glossy_transmission));
    EXPECT_FALSE(is_delta_surface_scattering_event(ScatteringLobe::specular));
    EXPECT_FALSE(is_delta_surface_scattering_event(ScatteringLobe::volume));
    EXPECT_FALSE(is_continuous_surface_scattering_event(KnownScatteringLobeMask));
}

TEST(TransportConventionsTest, DumpsTheExactVersionedBsdfContractDeterministically) {
    constexpr auto expected =
        R"json({"schema_version":1,"pdf":{"measures":{"discrete":0,"solid_angle":1,"area":2,"distance":3,"volume":4,"wavelength":5},"bsdf":{"conditional":"p(wi|wo)","reverse":"swap_wo_wi","continuous":"solid_angle","delta":"discrete","directional_query_excludes_delta":true,"projected_solid_angle":false,"component_selection":"included_exactly_once","eval_contains_cosine":false,"throughput":"f*abs(wi.z)/p"}},"lobe_flags":{"bits":{"none":0,"diffuse":1,"glossy":2,"specular":4,"reflection":8,"transmission":16,"volume":32},"known_mask":63,"surface_event":"exactly_one_family_and_one_direction","volume_event":"volume_only","selection_mask":"any_combination_of_known_bits","specular_is_delta":true},"transport_modes":{"radiance":{"code":0,"transmission_adjoint_scale":"(eta_i/eta_t)^2"},"importance":{"code":1,"transmission_adjoint_scale":"1"},"eta_i":"wo_side","eta_t":"wi_side","reflection":"mode_invariant"},"directions":{"space":"local_closure_frame","frame_basis":"caller_supplied","normal_axis":"+z","orientation":"away_from_surface","wo":"surface_to_previous_vertex","wi":"surface_to_next_vertex","reflection":"same_nonzero_hemisphere","transmission":"opposite_nonzero_hemispheres","tangent":"zero_support"}})json";

    const auto first = dump_bsdf_conventions(CurrentBsdfConventionSchemaVersion);
    const auto replay = dump_bsdf_conventions(CurrentBsdfConventionSchemaVersion);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    EXPECT_EQ(*first, expected);
    EXPECT_EQ(*replay, expected);
}

TEST(TransportConventionsTest, RejectsUnsupportedSchemaVersionsWithoutFallback) {
    for (const auto version : std::array{std::uint32_t{0U}, std::uint32_t{2U}}) {
        const auto dump = dump_bsdf_conventions(version);
        ASSERT_FALSE(dump.has_value());
        EXPECT_EQ(dump.error().code, core::StatusCode::incompatible);
        EXPECT_EQ(dump.error().message, "Unsupported BSDF convention schema version " +
                                            std::to_string(version) + "; expected 1.");
    }
}

} // namespace
} // namespace blackframe::renderer
