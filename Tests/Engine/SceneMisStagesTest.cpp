#include <Blackframe/Engine/SceneMisStages.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace blackframe::engine {
namespace {

inline constexpr auto TraceAddress = SceneMisPixelAddress{
    .sample =
        {
            .pixel_x = 3U,
            .pixel_y = 5U,
            .sample_index = 7U,
            .seed = 0x85A308D313198A2EULL,
        },
    .path_slot = {.value = std::numeric_limits<std::uint32_t>::max()},
};

[[nodiscard]] renderer::TransportSpectrum spectrum(const std::array<float, 4U> values) {
    return renderer::TransportSpectrum{.values = values};
}

[[nodiscard]] renderer::PathState initial_state() {
    return renderer::PathState::create_initial(
               renderer::sample_uniform_visible_wavelengths(0.25F).value(), renderer::VacuumMedium)
        .value();
}

[[nodiscard]] renderer::Ray primary_ray(const renderer::MediumId medium = renderer::VacuumMedium) {
    return renderer::Ray::create(renderer::Point3{.x = 0.0F, .y = -0.0F, .z = 2.0F},
                                 renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = -1.0F}, 0.0F,
                                 std::numeric_limits<float>::infinity(), 0.5F,
                                 renderer::RayMask{0xA5U}, medium)
        .value();
}

[[nodiscard]] renderer::Ray shadow_ray() {
    return renderer::Ray::create(renderer::Point3{}, renderer::Vector3{.y = 1.0F}, 0.0F, 4.0F, 0.5F,
                                 renderer::RayMask{0xA5U}, renderer::VacuumMedium)
        .value();
}

[[nodiscard]] renderer::Ray continuation_ray() {
    return renderer::Ray::create(renderer::Point3{.z = -0.01F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                                 std::numeric_limits<float>::infinity(), 0.5F,
                                 renderer::RayMask{0xA5U}, renderer::VacuumMedium)
        .value();
}

[[nodiscard]] AccelHit primary_hit() {
    return AccelHit{
        .object = {.value = 0x11U},
        .triangle =
            {
                .parameter = 2.0F,
                .position = {.x = 0.0F, .y = 0.0F, .z = 0.0F},
                .geometric_normal = {.z = 1.0F},
                .barycentrics = {.vertex0 = 0.25F, .vertex1 = 0.25F, .vertex2 = 0.5F},
            },
        .identifiers =
            {
                .instance = {.value = 0x22U},
                .geometry = {.value = 0x33U},
                .primitive = {.value = 0x44U},
                .material = {.value = 0x55U},
            },
    };
}

[[nodiscard]] SceneMisCameraStageOutput
camera_output(const SceneMisPixelAddress address = TraceAddress) {
    return SceneMisCameraStageOutput::create(address, primary_ray(), initial_state()).value();
}

[[nodiscard]] SceneMisIntersectionStageOutput
hit_output(const renderer::WavefrontPathSlot slot = TraceAddress.path_slot) {
    return SceneMisIntersectionStageOutput::hit(slot, primary_ray(), primary_hit()).value();
}

[[nodiscard]] SceneMisShadingStageOutput
shading_output(const bool requires_shadow = true, const bool continuation = false,
               const renderer::WavefrontPathSlot slot = TraceAddress.path_slot) {
    const auto requested_shadow = requires_shadow ? std::optional{shadow_ray()} : std::nullopt;
    const auto requested_continuation =
        continuation ? std::optional{continuation_ray()} : std::nullopt;
    const auto requested_continuation_throughput =
        continuation ? std::optional{initial_state().beta()} : std::nullopt;
    return SceneMisShadingStageOutput::create(
               slot, initial_state(), spectrum({0.125F, 0.25F, 0.5F, 1.0F}), requested_shadow,
               requested_continuation, requested_continuation_throughput)
        .value();
}

[[nodiscard]] SceneMisShadowStageOutput
visible_shadow_output(const renderer::WavefrontPathSlot slot = TraceAddress.path_slot) {
    return SceneMisShadowStageOutput::visible(slot, shadow_ray(),
                                              spectrum({1.0F, 2.0F, 3.0F, 4.0F}))
        .value();
}

[[nodiscard]] SceneMisAccumulationStageOutput terminated_output(
    const renderer::TransportSpectrum radiance = spectrum({1.125F, 2.25F, 3.5F, 5.0F}),
    const renderer::BsdfOnlyPathTermination termination =
        renderer::BsdfOnlyPathTermination::outside_bsdf_support,
    const renderer::WavefrontPathSlot slot = TraceAddress.path_slot) {
    return SceneMisAccumulationStageOutput::terminated(slot, radiance, termination).value();
}

[[nodiscard]] SceneMisPixelTrace complete_visible_trace() {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    EXPECT_TRUE(trace.append_camera(camera_output()).has_value());
    EXPECT_TRUE(trace.append_intersection(hit_output()).has_value());
    EXPECT_TRUE(trace.append_shading(shading_output()).has_value());
    EXPECT_TRUE(trace.append_shadow(visible_shadow_output()).has_value());
    EXPECT_TRUE(trace.append_accumulation(terminated_output()).has_value());
    return trace;
}

TEST(SceneMisStagesTest, SerializesDetailedVisiblePixelTraceByteForByte) {
    auto trace = complete_visible_trace();
    ASSERT_TRUE(trace.complete());
    ASSERT_EQ(trace.event_count(), 5U);
    constexpr auto expected_stages = std::array{
        SceneMisStageKind::camera, SceneMisStageKind::intersection, SceneMisStageKind::shading,
        SceneMisStageKind::shadow, SceneMisStageKind::accumulation,
    };
    for (auto index = std::size_t{}; index < expected_stages.size(); ++index) {
        ASSERT_TRUE(trace.stage_at(index).has_value());
        EXPECT_EQ(*trace.stage_at(index), expected_stages[index]);
    }
    EXPECT_FALSE(trace.stage_at(expected_stages.size()).has_value());

    const auto serialized = trace.serialize();
    ASSERT_TRUE(serialized.has_value()) << serialized.error().message;
    const auto expected = std::string{
        R"({"schema_version":1,"precision":"float32","pixel_x":"0x00000003","pixel_y":"0x00000005","sample_index":"0x0000000000000007","seed":"0x85a308d313198a2e","path_slot":"0xffffffff","events":[)"
        R"({"sequence":0,"stage":"camera","ray":{"origin_bits":["0x00000000","0x80000000","0x40000000"],"direction_bits":["0x00000000","0x00000000","0xbf800000"],"t_min_bits":"0x00000000","t_max_bits":"0x7f800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"initial_state":{"throughput_bits":["0x3f800000","0x3f800000","0x3f800000","0x3f800000"],"radiance_bits":["0x00000000","0x00000000","0x00000000","0x00000000"],"depth":"0x00000000","eta_scale_bits":"0x3f800000","delta_flags":"0x00","current_medium":"0x00000000"}},)"
        R"({"sequence":1,"stage":"intersection","ray":{"origin_bits":["0x00000000","0x80000000","0x40000000"],"direction_bits":["0x00000000","0x00000000","0xbf800000"],"t_min_bits":"0x00000000","t_max_bits":"0x7f800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"outcome":"hit","parameter_bits":"0x40000000","position_bits":["0x00000000","0x00000000","0x00000000"],"geometric_normal_bits":["0x00000000","0x00000000","0x3f800000"],"barycentrics_bits":["0x3e800000","0x3e800000","0x3f000000"],"object":"0x00000011","instance":"0x00000022","geometry":"0x00000033","primitive":"0x00000044","material":"0x00000055"},)"
        R"({"sequence":2,"stage":"shading","throughput_bits":["0x3f800000","0x3f800000","0x3f800000","0x3f800000"],"radiance_before_bits":["0x00000000","0x00000000","0x00000000","0x00000000"],"emitted_contribution_bits":["0x3e000000","0x3e800000","0x3f000000","0x3f800000"],"requires_shadow":true,"shadow_ray":{"origin_bits":["0x00000000","0x00000000","0x00000000"],"direction_bits":["0x00000000","0x3f800000","0x00000000"],"t_min_bits":"0x00000000","t_max_bits":"0x40800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"continuation":false},)"
        R"({"sequence":3,"stage":"shadow","ray":{"origin_bits":["0x00000000","0x00000000","0x00000000"],"direction_bits":["0x00000000","0x3f800000","0x00000000"],"t_min_bits":"0x00000000","t_max_bits":"0x40800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"outcome":"visible","direct_contribution_bits":["0x3f800000","0x40000000","0x40400000","0x40800000"]},)"
        R"({"sequence":4,"stage":"accumulation","radiance_bits":["0x3f900000","0x40100000","0x40600000","0x40a00000"],"route":"terminated","termination":"outside_bsdf_support"}]})"};
    EXPECT_EQ(*serialized, expected);
    EXPECT_EQ(*trace.serialize(), expected);
}

TEST(SceneMisStagesTest, TracesMissDirectlyToTerminalAccumulation) {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(trace
                    .append_intersection(
                        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray())
                            .value())
                    .has_value());
    ASSERT_TRUE(trace
                    .append_accumulation(
                        terminated_output(spectrum({0.5F, 0.5F, 0.5F, 0.5F}),
                                          renderer::BsdfOnlyPathTermination::escaped_environment))
                    .has_value());
    ASSERT_TRUE(trace.complete());
    ASSERT_EQ(trace.event_count(), 3U);
    EXPECT_EQ(*trace.stage_at(0U), SceneMisStageKind::camera);
    EXPECT_EQ(*trace.stage_at(1U), SceneMisStageKind::intersection);
    EXPECT_EQ(*trace.stage_at(2U), SceneMisStageKind::accumulation);

    const auto expected = std::string{
        R"({"schema_version":1,"precision":"float32","pixel_x":"0x00000003","pixel_y":"0x00000005","sample_index":"0x0000000000000007","seed":"0x85a308d313198a2e","path_slot":"0xffffffff","events":[)"
        R"({"sequence":0,"stage":"camera","ray":{"origin_bits":["0x00000000","0x80000000","0x40000000"],"direction_bits":["0x00000000","0x00000000","0xbf800000"],"t_min_bits":"0x00000000","t_max_bits":"0x7f800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"initial_state":{"throughput_bits":["0x3f800000","0x3f800000","0x3f800000","0x3f800000"],"radiance_bits":["0x00000000","0x00000000","0x00000000","0x00000000"],"depth":"0x00000000","eta_scale_bits":"0x3f800000","delta_flags":"0x00","current_medium":"0x00000000"}},)"
        R"({"sequence":1,"stage":"intersection","ray":{"origin_bits":["0x00000000","0x80000000","0x40000000"],"direction_bits":["0x00000000","0x00000000","0xbf800000"],"t_min_bits":"0x00000000","t_max_bits":"0x7f800000","time_bits":"0x3f000000","mask":"0x000000a5","current_medium":"0x00000000"},"outcome":"miss"},)"
        R"({"sequence":2,"stage":"accumulation","radiance_bits":["0x3f000000","0x3f000000","0x3f000000","0x3f000000"],"route":"terminated","termination":"escaped_environment"}]})"};
    ASSERT_TRUE(trace.serialize().has_value());
    EXPECT_EQ(*trace.serialize(), expected);
}

TEST(SceneMisStagesTest, SerializesOccludedShadowWithZeroDirectContribution) {
    auto visible = complete_visible_trace();
    auto occluded =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(occluded.append_camera(camera_output()).has_value());
    ASSERT_TRUE(occluded.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(occluded.append_shading(shading_output()).has_value());
    const auto shadow =
        SceneMisShadowStageOutput::occluded(TraceAddress.path_slot, shadow_ray()).value();
    EXPECT_EQ(shadow.direct_contribution(), renderer::TransportSpectrum{});
    ASSERT_TRUE(occluded.append_shadow(shadow).has_value());
    const auto dropped_contributions =
        occluded.append_accumulation(terminated_output(renderer::TransportSpectrum{}));
    ASSERT_FALSE(dropped_contributions.has_value());
    EXPECT_EQ(dropped_contributions.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(occluded.event_count(), 4U);
    ASSERT_TRUE(
        occluded.append_accumulation(terminated_output(spectrum({0.125F, 0.25F, 0.5F, 1.0F})))
            .has_value());

    const auto visible_bytes = visible.serialize().value();
    const auto occluded_bytes = occluded.serialize().value();
    const auto visible_marker = visible_bytes.find(R"("outcome":"visible")");
    const auto occluded_marker = occluded_bytes.find(R"("outcome":"occluded")");
    ASSERT_NE(visible_marker, std::string::npos);
    ASSERT_NE(occluded_marker, std::string::npos);
    EXPECT_EQ(visible_bytes.substr(0U, visible_marker), occluded_bytes.substr(0U, occluded_marker));
    EXPECT_NE(visible_bytes, occluded_bytes);
    EXPECT_NE(
        occluded_bytes.find(
            R"("direct_contribution_bits":["0x00000000","0x00000000","0x00000000","0x00000000"])"),
        std::string::npos);
    EXPECT_NE(occluded_bytes.find(
                  R"("radiance_bits":["0x3e000000","0x3e800000","0x3f000000","0x3f800000"])"),
              std::string::npos);
}

TEST(SceneMisStagesTest, RejectsOutOfOrderStagesWithoutMutatingOrPoisoningTrace) {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    const auto camera = camera_output();
    const auto intersection = hit_output();
    const auto shading = shading_output();
    const auto shadow = visible_shadow_output();
    const auto accumulated = terminated_output();

    EXPECT_FALSE(trace.append_intersection(intersection).has_value());
    EXPECT_EQ(trace.event_count(), 0U);
    ASSERT_TRUE(trace.append_camera(camera).has_value());
    EXPECT_FALSE(trace.append_camera(camera).has_value());
    EXPECT_FALSE(trace.append_shading(shading).has_value());
    EXPECT_EQ(trace.event_count(), 1U);
    ASSERT_TRUE(trace.append_intersection(intersection).has_value());
    EXPECT_FALSE(trace.append_shadow(shadow).has_value());
    EXPECT_FALSE(trace.append_accumulation(accumulated).has_value());
    EXPECT_EQ(trace.event_count(), 2U);
    ASSERT_TRUE(trace.append_shading(shading).has_value());
    EXPECT_FALSE(trace.append_accumulation(accumulated).has_value());
    EXPECT_EQ(trace.event_count(), 3U);

    const auto wrong_slot = renderer::WavefrontPathSlot{.value = TraceAddress.path_slot.value - 1U};
    const auto wrong_shadow = SceneMisShadowStageOutput::occluded(wrong_slot, shadow_ray()).value();
    const auto mismatch = trace.append_shadow(wrong_shadow);
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 3U);
    ASSERT_TRUE(trace.append_shadow(shadow).has_value());
    EXPECT_FALSE(trace.append_camera(camera).has_value());
    EXPECT_EQ(trace.event_count(), 4U);
    ASSERT_TRUE(trace.append_accumulation(accumulated).has_value());
    EXPECT_TRUE(trace.complete());
    EXPECT_FALSE(trace.append_accumulation(accumulated).has_value());
    EXPECT_EQ(trace.event_count(), 5U);
    EXPECT_EQ(*trace.serialize(), *complete_visible_trace().serialize());
}

TEST(SceneMisStagesTest, RejectsMissShadingShadowAndContinuationWithoutFallback) {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(trace
                    .append_intersection(
                        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray())
                            .value())
                    .has_value());
    EXPECT_FALSE(trace.append_shading(shading_output()).has_value());
    EXPECT_FALSE(trace.append_shadow(visible_shadow_output()).has_value());
    const auto continued =
        SceneMisAccumulationStageOutput::continuation(
            TraceAddress.path_slot, renderer::TransportSpectrum{}, continuation_ray())
            .value();
    EXPECT_FALSE(trace.append_accumulation(continued).has_value());
    EXPECT_EQ(trace.event_count(), 2U);
    EXPECT_FALSE(trace.complete());
    const auto incomplete = trace.serialize();
    ASSERT_FALSE(incomplete.has_value());
    EXPECT_EQ(incomplete.error().code, core::StatusCode::incompatible);

    ASSERT_TRUE(trace
                    .append_accumulation(
                        terminated_output(renderer::TransportSpectrum{},
                                          renderer::BsdfOnlyPathTermination::escaped_environment))
                    .has_value());
    EXPECT_TRUE(trace.complete());
}

TEST(SceneMisStagesTest, RoutesContinuationBackToIntersectionWithoutAHiddenStage) {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(trace.append_shading(shading_output(false, true)).has_value());
    const auto continued =
        SceneMisAccumulationStageOutput::continuation(
            TraceAddress.path_slot, spectrum({0.125F, 0.25F, 0.5F, 1.0F}), continuation_ray())
            .value();
    ASSERT_TRUE(trace.append_accumulation(continued).has_value());
    EXPECT_FALSE(trace.complete());
    EXPECT_EQ(trace.event_count(), 4U);
    EXPECT_FALSE(trace.append_camera(camera_output()).has_value());

    ASSERT_TRUE(trace
                    .append_intersection(SceneMisIntersectionStageOutput::miss(
                                             TraceAddress.path_slot, continuation_ray())
                                             .value())
                    .has_value());
    ASSERT_TRUE(trace
                    .append_accumulation(
                        terminated_output(spectrum({0.25F, 0.5F, 1.0F, 2.0F}),
                                          renderer::BsdfOnlyPathTermination::escaped_environment))
                    .has_value());
    EXPECT_TRUE(trace.complete());
    constexpr auto expected = std::array{
        SceneMisStageKind::camera,       SceneMisStageKind::intersection,
        SceneMisStageKind::shading,      SceneMisStageKind::accumulation,
        SceneMisStageKind::intersection, SceneMisStageKind::accumulation,
    };
    ASSERT_EQ(trace.event_count(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(*trace.stage_at(index), expected[index]);
    }
    const auto serialized = trace.serialize();
    ASSERT_TRUE(serialized.has_value());
    EXPECT_NE(serialized->find(R"("route":"continuation","termination":"none")"),
              std::string::npos);
    EXPECT_NE(serialized->find(
                  R"("continuation_ray":{"origin_bits":["0x00000000","0x00000000","0xbc23d70a"])"),
              std::string::npos);
    EXPECT_NE(
        serialized->find(
            R"("continuation_throughput_bits":["0x3f800000","0x3f800000","0x3f800000","0x3f800000"])"),
        std::string::npos);
}

TEST(SceneMisStagesTest, RejectsUnroutedRaysWithoutMutatingTheTrace) {
    auto trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(trace.append_camera(camera_output()).has_value());

    const auto sign_changed_ray =
        renderer::Ray::create(renderer::Point3{.x = 0.0F, .y = 0.0F, .z = 2.0F},
                              renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<float>::infinity(), 0.5F,
                              renderer::RayMask{0xA5U}, renderer::VacuumMedium)
            .value();
    const auto sign_changed_intersection =
        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, sign_changed_ray).value();
    const auto sign_status = trace.append_intersection(sign_changed_intersection);
    ASSERT_FALSE(sign_status.has_value());
    EXPECT_EQ(sign_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 1U);

    const auto wrong_intersection =
        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, continuation_ray()).value();
    const auto intersection_status = trace.append_intersection(wrong_intersection);
    ASSERT_FALSE(intersection_status.has_value());
    EXPECT_EQ(intersection_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 1U);

    ASSERT_TRUE(trace.append_intersection(hit_output()).has_value());
    const auto half_beta = spectrum({0.5F, 0.5F, 0.5F, 0.5F});
    const auto mismatched_beta_state =
        renderer::PathState::create(half_beta, renderer::TransportSpectrum{}, {}, 1.0F,
                                    initial_state().wavelengths(), renderer::PathDeltaFlags::none,
                                    renderer::VacuumMedium)
            .value();
    const auto mismatched_beta_shading =
        SceneMisShadingStageOutput::create(TraceAddress.path_slot, mismatched_beta_state,
                                           renderer::TransportSpectrum{}, shadow_ray(),
                                           continuation_ray(), half_beta)
            .value();
    const auto beta_status = trace.append_shading(mismatched_beta_shading);
    ASSERT_FALSE(beta_status.has_value());
    EXPECT_EQ(beta_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 2U);

    const auto mismatched_state =
        renderer::PathState::create(initial_state().beta(), spectrum({1.0F, 1.0F, 1.0F, 1.0F}), {},
                                    1.0F, initial_state().wavelengths(),
                                    renderer::PathDeltaFlags::none, renderer::VacuumMedium)
            .value();
    const auto mismatched_shading =
        SceneMisShadingStageOutput::create(TraceAddress.path_slot, mismatched_state,
                                           renderer::TransportSpectrum{}, shadow_ray(),
                                           continuation_ray(), initial_state().beta())
            .value();
    const auto shading_status = trace.append_shading(mismatched_shading);
    ASSERT_FALSE(shading_status.has_value());
    EXPECT_EQ(shading_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 2U);

    auto signed_zero_radiance = renderer::TransportSpectrum{};
    signed_zero_radiance[0U] = -0.0F;
    const auto signed_zero_state =
        renderer::PathState::create(initial_state().beta(), signed_zero_radiance, {}, 1.0F,
                                    initial_state().wavelengths(), renderer::PathDeltaFlags::none,
                                    renderer::VacuumMedium)
            .value();
    const auto signed_zero_shading =
        SceneMisShadingStageOutput::create(TraceAddress.path_slot, signed_zero_state,
                                           renderer::TransportSpectrum{}, shadow_ray(),
                                           continuation_ray(), initial_state().beta())
            .value();
    const auto signed_zero_status = trace.append_shading(signed_zero_shading);
    ASSERT_FALSE(signed_zero_status.has_value());
    EXPECT_EQ(signed_zero_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 2U);

    const auto wrong_mask_shadow =
        renderer::Ray::create(renderer::Point3{}, renderer::Vector3{.y = 1.0F}, 0.0F, 4.0F, 0.5F,
                              renderer::RayMask{0x0FU}, renderer::VacuumMedium)
            .value();
    const auto wrong_mask_shading =
        SceneMisShadingStageOutput::create(TraceAddress.path_slot, initial_state(),
                                           renderer::TransportSpectrum{}, wrong_mask_shadow,
                                           continuation_ray(), initial_state().beta())
            .value();
    const auto mask_status = trace.append_shading(wrong_mask_shading);
    ASSERT_FALSE(mask_status.has_value());
    EXPECT_EQ(mask_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 2U);
    ASSERT_TRUE(trace.append_shading(shading_output(true, true)).has_value());
    const auto wrong_shadow =
        SceneMisShadowStageOutput::occluded(TraceAddress.path_slot, primary_ray()).value();
    const auto shadow_status = trace.append_shadow(wrong_shadow);
    ASSERT_FALSE(shadow_status.has_value());
    EXPECT_EQ(shadow_status.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 3U);

    ASSERT_TRUE(
        trace
            .append_shadow(
                SceneMisShadowStageOutput::occluded(TraceAddress.path_slot, shadow_ray()).value())
            .has_value());
    ASSERT_TRUE(
        trace
            .append_accumulation(SceneMisAccumulationStageOutput::continuation(
                                     TraceAddress.path_slot, spectrum({0.125F, 0.25F, 0.5F, 1.0F}),
                                     continuation_ray())
                                     .value())
            .has_value());

    const auto wrong_bounce = trace.append_intersection(
        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray()).value());
    ASSERT_FALSE(wrong_bounce.has_value());
    EXPECT_EQ(wrong_bounce.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(trace.event_count(), 5U);
    EXPECT_FALSE(trace.complete());
}

TEST(SceneMisStagesTest, RejectsTerminationRoutesThatContradictThePrecedingStage) {
    auto miss_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(miss_trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(miss_trace
                    .append_intersection(
                        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray())
                            .value())
                    .has_value());
    auto invalid = miss_trace.append_accumulation(terminated_output());
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(miss_trace.event_count(), 2U);

    const auto radiance_state =
        renderer::PathState::create(initial_state().beta(), spectrum({1.0F, 2.0F, 3.0F, 4.0F}), {},
                                    1.0F, initial_state().wavelengths(),
                                    renderer::PathDeltaFlags::none, renderer::VacuumMedium)
            .value();
    auto monotonic_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(monotonic_trace
                    .append_camera(SceneMisCameraStageOutput::create(TraceAddress, primary_ray(),
                                                                     radiance_state)
                                       .value())
                    .has_value());
    ASSERT_TRUE(monotonic_trace
                    .append_intersection(
                        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray())
                            .value())
                    .has_value());
    const auto reduced_environment = monotonic_trace.append_accumulation(terminated_output(
        renderer::TransportSpectrum{}, renderer::BsdfOnlyPathTermination::escaped_environment));
    ASSERT_FALSE(reduced_environment.has_value());
    EXPECT_EQ(reduced_environment.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(monotonic_trace.event_count(), 2U);

    auto hit_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(hit_trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(hit_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(hit_trace.append_shading(shading_output(false, false)).has_value());
    invalid = hit_trace.append_accumulation(
        terminated_output({}, renderer::BsdfOnlyPathTermination::escaped_environment));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    invalid = hit_trace.append_accumulation(
        terminated_output({}, renderer::BsdfOnlyPathTermination::russian_roulette));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    invalid = hit_trace.append_accumulation(terminated_output(
        spectrum({0.125F, 0.25F, 0.5F, 1.0F}), renderer::BsdfOnlyPathTermination::zero_throughput));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(hit_trace.event_count(), 3U);

    auto shadow_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(shadow_trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(shadow_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(shadow_trace.append_shading(shading_output(true, false)).has_value());
    ASSERT_TRUE(shadow_trace.append_shadow(visible_shadow_output()).has_value());
    invalid = shadow_trace.append_accumulation(terminated_output(
        spectrum({1.125F, 2.25F, 3.5F, 5.0F}), renderer::BsdfOnlyPathTermination::depth_limit));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(shadow_trace.event_count(), 4U);
    EXPECT_TRUE(shadow_trace.append_accumulation(terminated_output()).has_value());
    EXPECT_TRUE(shadow_trace.complete());

    const auto zero_state =
        renderer::PathState::create(renderer::TransportSpectrum{}, renderer::TransportSpectrum{},
                                    {}, 1.0F, initial_state().wavelengths(),
                                    renderer::PathDeltaFlags::none, renderer::VacuumMedium)
            .value();
    auto zero_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(
        zero_trace
            .append_camera(
                SceneMisCameraStageOutput::create(TraceAddress, primary_ray(), zero_state).value())
            .has_value());
    ASSERT_TRUE(zero_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(zero_trace
                    .append_shading(
                        SceneMisShadingStageOutput::create(TraceAddress.path_slot, zero_state,
                                                           renderer::TransportSpectrum{},
                                                           std::nullopt, std::nullopt, std::nullopt)
                            .value())
                    .has_value());
    invalid = zero_trace.append_accumulation(
        terminated_output({}, renderer::BsdfOnlyPathTermination::outside_bsdf_support));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_TRUE(zero_trace
                    .append_accumulation(
                        terminated_output({}, renderer::BsdfOnlyPathTermination::zero_throughput))
                    .has_value());
    EXPECT_TRUE(zero_trace.complete());

    auto prepared_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(prepared_trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(prepared_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(prepared_trace.append_shading(shading_output(false, true)).has_value());
    invalid = prepared_trace.append_accumulation(terminated_output());
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(prepared_trace.event_count(), 3U);
    EXPECT_TRUE(prepared_trace
                    .append_accumulation(
                        terminated_output(spectrum({0.125F, 0.25F, 0.5F, 1.0F}),
                                          renderer::BsdfOnlyPathTermination::russian_roulette))
                    .has_value());
    EXPECT_TRUE(prepared_trace.complete());

    auto zero_continuation_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(zero_continuation_trace.append_camera(camera_output()).has_value());
    ASSERT_TRUE(zero_continuation_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(zero_continuation_trace
                    .append_shading(SceneMisShadingStageOutput::create(
                                        TraceAddress.path_slot, initial_state(),
                                        spectrum({0.125F, 0.25F, 0.5F, 1.0F}), std::nullopt,
                                        continuation_ray(), renderer::TransportSpectrum{})
                                        .value())
                    .has_value());
    invalid = zero_continuation_trace.append_accumulation(
        SceneMisAccumulationStageOutput::continuation(
            TraceAddress.path_slot, spectrum({0.125F, 0.25F, 0.5F, 1.0F}), continuation_ray())
            .value());
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    invalid = zero_continuation_trace.append_accumulation(
        terminated_output(spectrum({0.125F, 0.25F, 0.5F, 1.0F}),
                          renderer::BsdfOnlyPathTermination::russian_roulette));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_TRUE(zero_continuation_trace
                    .append_accumulation(
                        terminated_output(spectrum({0.125F, 0.25F, 0.5F, 1.0F}),
                                          renderer::BsdfOnlyPathTermination::zero_throughput))
                    .has_value());
    EXPECT_TRUE(zero_continuation_trace.complete());
}

TEST(SceneMisStagesTest, RejectsUnsupportedSchemaAndMalformedStageOutputs) {
    for (const auto version : {0U, CurrentSceneMisStageTraceSchemaVersion + 1U,
                               std::numeric_limits<std::uint32_t>::max()}) {
        const auto trace = SceneMisPixelTrace::create(TraceAddress, version);
        ASSERT_FALSE(trace.has_value());
        EXPECT_EQ(trace.error().code, core::StatusCode::incompatible);
        EXPECT_FALSE(trace.error().message.empty());
    }

    auto beta = spectrum({1.0F, 1.0F, 1.0F, 1.0F});
    beta[0U] = -1.0F;
    const auto signed_state =
        renderer::PathState::create(beta, renderer::TransportSpectrum{}, {}, 1.0F,
                                    initial_state().wavelengths(), renderer::PathDeltaFlags::none,
                                    renderer::VacuumMedium)
            .value();
    EXPECT_FALSE(
        SceneMisCameraStageOutput::create(TraceAddress, primary_ray(), signed_state).has_value());
    EXPECT_FALSE(SceneMisCameraStageOutput::create(
                     TraceAddress, primary_ray(renderer::MediumId{.value = 9U}), initial_state())
                     .has_value());

    auto inconsistent_hit = primary_hit();
    inconsistent_hit.triangle.position.z = 1.0F;
    EXPECT_FALSE(SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(),
                                                      inconsistent_hit)
                     .has_value());
    auto signed_zero_hit = primary_hit();
    signed_zero_hit.triangle.position.y = -0.0F;
    EXPECT_FALSE(
        SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(), signed_zero_hit)
            .has_value());

    auto invalid_spectrum = renderer::TransportSpectrum{};
    invalid_spectrum[2U] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(SceneMisShadingStageOutput::create(TraceAddress.path_slot, initial_state(),
                                                    invalid_spectrum, shadow_ray(), std::nullopt,
                                                    std::nullopt)
                     .has_value());
    EXPECT_FALSE(SceneMisShadingStageOutput::create(TraceAddress.path_slot, initial_state(),
                                                    renderer::TransportSpectrum{}, std::nullopt,
                                                    continuation_ray(), std::nullopt)
                     .has_value());
    const auto zero_beta_state =
        renderer::PathState::create(renderer::TransportSpectrum{}, renderer::TransportSpectrum{},
                                    {}, 1.0F, initial_state().wavelengths(),
                                    renderer::PathDeltaFlags::none, renderer::VacuumMedium)
            .value();
    EXPECT_FALSE(SceneMisShadingStageOutput::create(TraceAddress.path_slot, zero_beta_state,
                                                    renderer::TransportSpectrum{}, shadow_ray(),
                                                    std::nullopt, std::nullopt)
                     .has_value());

    const auto partial_beta = spectrum({0.0F, 1.0F, 1.0F, 1.0F});
    const auto partial_state =
        renderer::PathState::create(partial_beta, renderer::TransportSpectrum{}, {}, 1.0F,
                                    initial_state().wavelengths(), renderer::PathDeltaFlags::none,
                                    renderer::VacuumMedium)
            .value();
    EXPECT_FALSE(SceneMisShadingStageOutput::create(TraceAddress.path_slot, partial_state,
                                                    spectrum({1.0F, 0.0F, 0.0F, 0.0F}),
                                                    std::nullopt, std::nullopt, std::nullopt)
                     .has_value());
    EXPECT_FALSE(SceneMisShadingStageOutput::create(TraceAddress.path_slot, partial_state,
                                                    renderer::TransportSpectrum{}, std::nullopt,
                                                    continuation_ray(), initial_state().beta())
                     .has_value());

    auto partial_shadow_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(partial_shadow_trace
                    .append_camera(SceneMisCameraStageOutput::create(TraceAddress, primary_ray(),
                                                                     partial_state)
                                       .value())
                    .has_value());
    ASSERT_TRUE(partial_shadow_trace.append_intersection(hit_output()).has_value());
    ASSERT_TRUE(partial_shadow_trace
                    .append_shading(
                        SceneMisShadingStageOutput::create(TraceAddress.path_slot, partial_state,
                                                           renderer::TransportSpectrum{},
                                                           shadow_ray(), std::nullopt, std::nullopt)
                            .value())
                    .has_value());
    const auto revived_direct = partial_shadow_trace.append_shadow(
        SceneMisShadowStageOutput::visible(TraceAddress.path_slot, shadow_ray(),
                                           spectrum({1.0F, 0.0F, 0.0F, 0.0F}))
            .value());
    ASSERT_FALSE(revived_direct.has_value());
    EXPECT_EQ(revived_direct.error().code, core::StatusCode::invalid_argument);

    auto partial_miss_trace =
        SceneMisPixelTrace::create(TraceAddress, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(partial_miss_trace
                    .append_camera(SceneMisCameraStageOutput::create(TraceAddress, primary_ray(),
                                                                     partial_state)
                                       .value())
                    .has_value());
    ASSERT_TRUE(partial_miss_trace
                    .append_intersection(
                        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray())
                            .value())
                    .has_value());
    const auto revived_environment = partial_miss_trace.append_accumulation(
        terminated_output(spectrum({1.0F, 0.0F, 0.0F, 0.0F}),
                          renderer::BsdfOnlyPathTermination::escaped_environment));
    ASSERT_FALSE(revived_environment.has_value());
    EXPECT_EQ(revived_environment.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(
        SceneMisShadowStageOutput::visible(TraceAddress.path_slot, shadow_ray(), invalid_spectrum)
            .has_value());
    EXPECT_FALSE(SceneMisAccumulationStageOutput::terminated(
                     TraceAddress.path_slot, renderer::TransportSpectrum{},
                     static_cast<renderer::BsdfOnlyPathTermination>(255U))
                     .has_value());

    auto invalid_barycentrics = primary_hit();
    invalid_barycentrics.triangle.barycentrics = {
        .vertex0 = 2.0F, .vertex1 = -1.0F, .vertex2 = 0.0F};
    EXPECT_FALSE(SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(),
                                                      invalid_barycentrics)
                     .has_value());
    invalid_barycentrics.triangle.barycentrics = {
        .vertex0 = 0.1F, .vertex1 = 0.1F, .vertex2 = 0.1F};
    EXPECT_FALSE(SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(),
                                                      invalid_barycentrics)
                     .has_value());

    auto invalid_normal = primary_hit();
    invalid_normal.triangle.geometric_normal.z = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(
        SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(), invalid_normal)
            .has_value());
    invalid_normal = primary_hit();
    invalid_normal.triangle.geometric_normal.z = 2.0F;
    EXPECT_FALSE(
        SceneMisIntersectionStageOutput::hit(TraceAddress.path_slot, primary_ray(), invalid_normal)
            .has_value());

    const auto medium = renderer::MediumId{.value = 9U};
    const auto medium_state =
        renderer::PathState::create_initial(initial_state().wavelengths(), medium).value();
    const auto medium_camera =
        SceneMisCameraStageOutput::create(TraceAddress, primary_ray(medium), medium_state);
    ASSERT_FALSE(medium_camera.has_value());
    EXPECT_EQ(medium_camera.error().code, core::StatusCode::unavailable);
    const auto medium_intersection =
        SceneMisIntersectionStageOutput::miss(TraceAddress.path_slot, primary_ray(medium));
    ASSERT_FALSE(medium_intersection.has_value());
    EXPECT_EQ(medium_intersection.error().code, core::StatusCode::unavailable);
    const auto medium_shadow =
        SceneMisShadowStageOutput::occluded(TraceAddress.path_slot, primary_ray(medium));
    ASSERT_FALSE(medium_shadow.has_value());
    EXPECT_EQ(medium_shadow.error().code, core::StatusCode::unavailable);

    const auto out_of_shutter_ray =
        renderer::Ray::create(renderer::Point3{.z = 2.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<float>::infinity(), 2.0F,
                              renderer::AllRayVisibility, renderer::VacuumMedium)
            .value();
    const auto invalid_time =
        SceneMisCameraStageOutput::create(TraceAddress, out_of_shutter_ray, initial_state());
    ASSERT_FALSE(invalid_time.has_value());
    EXPECT_EQ(invalid_time.error().code, core::StatusCode::invalid_argument);
}

TEST(SceneMisStagesTest, KeepsStageNamesAndIndependentTraceReplayStable) {
    EXPECT_STREQ(scene_mis_stage_name(SceneMisStageKind::camera), "camera");
    EXPECT_STREQ(scene_mis_stage_name(SceneMisStageKind::intersection), "intersection");
    EXPECT_STREQ(scene_mis_stage_name(SceneMisStageKind::shading), "shading");
    EXPECT_STREQ(scene_mis_stage_name(SceneMisStageKind::shadow), "shadow");
    EXPECT_STREQ(scene_mis_stage_name(SceneMisStageKind::accumulation), "accumulation");
    EXPECT_STREQ(scene_mis_stage_name(static_cast<SceneMisStageKind>(255U)), "unknown");

    auto first = complete_visible_trace();
    const auto other_address = SceneMisPixelAddress{
        .sample = {.pixel_x = 99U, .pixel_y = 77U, .sample_index = 55U, .seed = 33U},
        .path_slot = {.value = 11U},
    };
    auto unrelated =
        SceneMisPixelTrace::create(other_address, CurrentSceneMisStageTraceSchemaVersion).value();
    ASSERT_TRUE(unrelated.append_camera(camera_output(other_address)).has_value());
    auto replay = complete_visible_trace();

    const auto first_bytes = first.serialize();
    const auto replay_bytes = replay.serialize();
    ASSERT_TRUE(first_bytes.has_value());
    ASSERT_TRUE(replay_bytes.has_value());
    EXPECT_EQ(*first_bytes, *replay_bytes);
    EXPECT_FALSE(unrelated.complete());
    EXPECT_EQ(unrelated.event_count(), 1U);
}

} // namespace
} // namespace blackframe::engine
