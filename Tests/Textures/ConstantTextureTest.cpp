#include <Blackframe/Renderer/ConstantTexture.hpp>
#include <array>
#include <bit>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(ConstantTextureTest, KeepsTheThreeValueDomainsDistinctAndStable) {
    static_assert(!std::is_same_v<ConstantFloatTexture, ConstantColorTexture>);
    static_assert(!std::is_same_v<ConstantFloatTexture, ConstantSpectrumTexture>);
    static_assert(!std::is_same_v<ConstantColorTexture, ConstantSpectrumTexture>);
    static_assert(ConstantFloatTexture::kind() == ConstantTextureKind::float_value);
    static_assert(ConstantColorTexture::kind() == ConstantTextureKind::color);
    static_assert(ConstantSpectrumTexture::kind() == ConstantTextureKind::spectrum);

    EXPECT_EQ(ConstantFloatTexture{}.value(), 0.0F);
    EXPECT_EQ(ConstantColorTexture{}.value(), LinearRGB{});
    EXPECT_EQ(ConstantSpectrumTexture{}.value(), TransportSpectrum{});
}

TEST(ConstantTextureTest, PreservesFiniteSignedValuesWithoutClamping) {
    const auto scalar = ConstantFloatTexture::create(-17.25F);
    const auto color =
        ConstantColorTexture::create(LinearRGB{.red = -2.0F, .green = 3.5F, .blue = 0x1p100F});
    const auto spectrum = ConstantSpectrumTexture::create(
        TransportSpectrum{.values = {-4.0F, -0.0F, 7.25F, 0x1p100F}});

    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    ASSERT_TRUE(color.has_value()) << color.error().message;
    ASSERT_TRUE(spectrum.has_value()) << spectrum.error().message;
    EXPECT_EQ(scalar->value(), -17.25F);
    EXPECT_EQ(color->value(), (LinearRGB{.red = -2.0F, .green = 3.5F, .blue = 0x1p100F}));
    EXPECT_EQ(spectrum->value(), (TransportSpectrum{.values = {-4.0F, -0.0F, 7.25F, 0x1p100F}}));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(spectrum->value()[1U]),
              std::bit_cast<std::uint32_t>(-0.0F));

    const auto positive_zero = ConstantFloatTexture::create(0.0F).value();
    const auto negative_zero = ConstantFloatTexture::create(-0.0F).value();
    EXPECT_NE(positive_zero, negative_zero);
}

TEST(ConstantTextureTest, RejectsEveryNonFinitePayloadExplicitly) {
    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();

    for (const auto value : {infinity, -infinity, nan}) {
        const auto scalar = ConstantFloatTexture::create(value);
        EXPECT_FALSE(scalar.has_value());
        if (!scalar) {
            EXPECT_EQ(scalar.error().code, core::StatusCode::invalid_argument);
            EXPECT_FALSE(scalar.error().message.empty());
        }
    }

    const auto colors = std::array{
        LinearRGB{.red = infinity, .green = 2.0F, .blue = 3.0F},
        LinearRGB{.red = 1.0F, .green = infinity, .blue = 3.0F},
        LinearRGB{.red = 1.0F, .green = 2.0F, .blue = infinity},
    };
    for (const auto value : colors) {
        const auto color = ConstantColorTexture::create(value);
        ASSERT_FALSE(color.has_value());
        EXPECT_EQ(color.error().code, core::StatusCode::invalid_argument);
    }

    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        auto value = TransportSpectrum{.values = {1.0F, 2.0F, 3.0F, 4.0F}};
        value[lane] = nan;
        const auto spectrum = ConstantSpectrumTexture::create(value);
        ASSERT_FALSE(spectrum.has_value());
        EXPECT_EQ(spectrum.error().code, core::StatusCode::invalid_argument);
    }
}

TEST(ConstantTextureTest, WidensEachStoredFloatExactlyToReferencePrecision) {
    const auto scalar = ConstantFloatTexture::create(-0.1F).value();
    const auto color =
        ConstantColorTexture::create(LinearRGB{.red = -0.25F, .green = 0.5F, .blue = 12.75F})
            .value();
    const auto spectrum =
        ConstantSpectrumTexture::create(TransportSpectrum{.values = {-8.0F, 0.125F, 1.5F, 99.0F}})
            .value();

    EXPECT_EQ(widen_constant_texture_value(scalar), static_cast<ReferenceScalar>(scalar.value()));
    EXPECT_EQ(widen_constant_texture_value(color),
              (ReferenceLinearRGB{
                  .red = static_cast<ReferenceScalar>(color.value().red),
                  .green = static_cast<ReferenceScalar>(color.value().green),
                  .blue = static_cast<ReferenceScalar>(color.value().blue),
              }));

    const auto widened_spectrum = widen_constant_texture_value(spectrum);
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ(widened_spectrum[lane], static_cast<ReferenceScalar>(spectrum.value()[lane]));
    }
}

} // namespace
} // namespace blackframe::renderer
