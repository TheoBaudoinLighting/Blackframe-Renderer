#include <Blackframe/Engine/ConstantTextureEvaluation.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/ConstantTexture.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] FrameSceneDescription make_constant_texture_scene_description() {
    const auto scalar = renderer::ConstantFloatTexture::create(-1.25F).value();
    const auto color = renderer::ConstantColorTexture::create(
                           renderer::LinearRGB{.red = -0.5F, .green = 0.25F, .blue = 3.0F})
                           .value();
    const auto spectrum =
        renderer::ConstantSpectrumTexture::create(renderer::TransportSpectrum{
                                                      .values = {-2.0F, 0.125F, 4.5F, 8.0F},
                                                  })
            .value();
    return FrameSceneDescription{
        .constant_textures =
            {
                SceneConstantTexture{.id = {.value = 30U}, .texture = spectrum},
                SceneConstantTexture{.id = {.value = 10U}, .texture = scalar},
                SceneConstantTexture{.id = {.value = 20U}, .texture = color},
            },
        .objects = {},
        .geometries = {},
        .materials = {},
        .instances = {},
        .punctual_lights = {},
        .spectral_environment = std::nullopt,
    };
}

TEST(ConstantTextureFrameSceneTest, ClosesTheRegistryInStableIdentifierOrder) {
    const auto scene = FrameScene::create(make_constant_texture_scene_description());

    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto textures = (*scene)->constant_textures();
    ASSERT_EQ(textures.size(), 3U);
    EXPECT_EQ(textures[0].id.value, 10U);
    EXPECT_EQ(textures[0].kind(), renderer::ConstantTextureKind::float_value);
    EXPECT_EQ(textures[1].id.value, 20U);
    EXPECT_EQ(textures[1].kind(), renderer::ConstantTextureKind::color);
    EXPECT_EQ(textures[2].id.value, 30U);
    EXPECT_EQ(textures[2].kind(), renderer::ConstantTextureKind::spectrum);

    const auto color = (*scene)->constant_texture(renderer::TextureId{.value = 20U});
    ASSERT_TRUE(color.has_value()) << color.error().message;
    EXPECT_EQ(color->get(), textures[1]);
}

TEST(ConstantTextureFrameSceneTest, RejectsDuplicateIdentifiersBeforePublication) {
    auto description = make_constant_texture_scene_description();
    description.constant_textures.push_back(SceneConstantTexture{
        .id = {.value = 20U},
        .texture = renderer::ConstantFloatTexture::create(7.0F).value(),
    });

    const auto scene = FrameScene::create(std::move(description));

    ASSERT_FALSE(scene.has_value());
    EXPECT_EQ(scene.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(scene.error().message.empty());
}

TEST(ConstantTextureEvaluationTest, MatchesScalarReferenceAndCpuTransportExactly) {
    const auto scene = FrameScene::create(make_constant_texture_scene_description());
    ASSERT_TRUE(scene.has_value()) << scene.error().message;

    const auto cpu_float =
        evaluate_cpu_transport_constant_float_texture(**scene, renderer::TextureId{.value = 10U});
    const auto cpu_color =
        evaluate_cpu_transport_constant_color_texture(**scene, renderer::TextureId{.value = 20U});
    const auto cpu_spectrum = evaluate_cpu_transport_constant_spectrum_texture(
        **scene, renderer::TextureId{.value = 30U});
    const auto reference_float = evaluate_scalar_reference_constant_float_texture(
        **scene, renderer::TextureId{.value = 10U});
    const auto reference_color = evaluate_scalar_reference_constant_color_texture(
        **scene, renderer::TextureId{.value = 20U});
    const auto reference_spectrum = evaluate_scalar_reference_constant_spectrum_texture(
        **scene, renderer::TextureId{.value = 30U});

    ASSERT_TRUE(cpu_float.has_value()) << cpu_float.error().message;
    ASSERT_TRUE(cpu_color.has_value()) << cpu_color.error().message;
    ASSERT_TRUE(cpu_spectrum.has_value()) << cpu_spectrum.error().message;
    ASSERT_TRUE(reference_float.has_value()) << reference_float.error().message;
    ASSERT_TRUE(reference_color.has_value()) << reference_color.error().message;
    ASSERT_TRUE(reference_spectrum.has_value()) << reference_spectrum.error().message;
    EXPECT_EQ(*cpu_float, -1.25F);
    EXPECT_EQ(*cpu_color, (renderer::LinearRGB{.red = -0.5F, .green = 0.25F, .blue = 3.0F}));
    EXPECT_EQ(*cpu_spectrum, (renderer::TransportSpectrum{.values = {-2.0F, 0.125F, 4.5F, 8.0F}}));
    EXPECT_EQ(*reference_float, static_cast<renderer::ReferenceScalar>(*cpu_float));
    EXPECT_EQ(*reference_color,
              (renderer::ReferenceLinearRGB{
                  .red = static_cast<renderer::ReferenceScalar>(cpu_color->red),
                  .green = static_cast<renderer::ReferenceScalar>(cpu_color->green),
                  .blue = static_cast<renderer::ReferenceScalar>(cpu_color->blue),
              }));
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ((*reference_spectrum)[lane],
                  static_cast<renderer::ReferenceScalar>((*cpu_spectrum)[lane]));
    }
}

TEST(ConstantTextureEvaluationTest, RejectsMissingIdentifiersAndKindMismatches) {
    const auto scene = FrameScene::create(make_constant_texture_scene_description());
    ASSERT_TRUE(scene.has_value()) << scene.error().message;

    const auto missing =
        evaluate_cpu_transport_constant_float_texture(**scene, renderer::TextureId{.value = 404U});
    const auto cpu_mismatch = evaluate_cpu_transport_constant_spectrum_texture(
        **scene, renderer::TextureId{.value = 20U});
    const auto reference_mismatch = evaluate_scalar_reference_constant_color_texture(
        **scene, renderer::TextureId{.value = 10U});

    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);
    ASSERT_FALSE(cpu_mismatch.has_value());
    EXPECT_EQ(cpu_mismatch.error().code, core::StatusCode::incompatible);
    ASSERT_FALSE(reference_mismatch.has_value());
    EXPECT_EQ(reference_mismatch.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(cpu_mismatch.error().message.empty());
    EXPECT_FALSE(reference_mismatch.error().message.empty());
}

} // namespace
} // namespace blackframe::engine
