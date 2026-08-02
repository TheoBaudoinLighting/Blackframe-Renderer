#include "CornellWavefrontScene.hpp"
#include "ScalarWavefrontImageParity.hpp"

#include <Blackframe/Renderer/PixelJitter.hpp>
#include <cstdint>
#include <gtest/gtest.h>

namespace blackframe::engine {
namespace {

TEST(CornellWavefrontParityTest, MatchesScalarReferenceThroughCpuWavefront) {
    constexpr auto extent = renderer::RenderExtent{.width = 4U, .height = 4U};
    constexpr auto samples_per_pixel = std::uint32_t{2U};
    constexpr auto seed = std::uint64_t{0x243F6A8885A308D3ULL};
    constexpr auto path_time = renderer::TransportScalar{0.5F};

    const auto scene = cornell_wavefront_test::make_cornell_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->geometries().size(), 7U);
    ASSERT_EQ((*scene)->instances().size(), 8U);
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 1U);
    ASSERT_TRUE((*scene)->spectral_environment().has_value());

    const auto camera = cornell_wavefront_test::make_camera(extent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = scalar_wavefront_parity_test::make_inputs(
        extent, samples_per_pixel, seed, (*scene)->spectral_environment()->wavelengths,
        [&camera](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return camera->generate_primary_ray(index, renderer::PixelJitterMode::uniform,
                                                path_time);
        });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto configuration = scalar_wavefront_parity_test::Configuration{
        .scene_name = "CornellDiffuse",
        .extent = extent,
        .samples_per_pixel = samples_per_pixel,
        .seed = seed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 5U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
        .worker_count = 4U,
        .thresholds = scalar_wavefront_parity_test::StrictThresholds,
    };
    const auto parity = scalar_wavefront_parity_test::compare(*scene, *inputs, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    EXPECT_GT(parity->wavefront_report.closure_samples, 0U);
    EXPECT_GT(parity->wavefront_report.light_samples, 0U);
    EXPECT_GT(parity->wavefront_report.shadow_queries, 0U);
    scalar_wavefront_parity_test::record_and_expect(configuration, *parity);
}

} // namespace
} // namespace blackframe::engine
