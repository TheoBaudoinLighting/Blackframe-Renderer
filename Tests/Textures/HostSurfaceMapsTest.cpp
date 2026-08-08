#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <Blackframe/Renderer/HostSurfaceMaps.hpp>
#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::filesystem::path surface_map_artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::filesystem::path
write_pfm_fixture(const std::string_view name, const std::uint32_t width,
                  const std::uint32_t height, const std::uint32_t channel_count,
                  const std::span<const TransportScalar> pixels) {
    const auto output = surface_map_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_TRUE(channel_count == 1U || channel_count == 3U);
    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(width) * height * channel_count);
    stream << (channel_count == 1U ? "Pf\n" : "PF\n") << width << ' ' << height << '\n'
           << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size_bytes()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] HostImageMipChainHandle
load_surface_map(HostImageCache& cache, const std::filesystem::path& path,
                 const TextureColorSpace color_space = TextureColorSpace::data) {
    const auto image = cache.load(path, color_space);
    EXPECT_TRUE(image.has_value()) << image.error().message;
    if (!image) {
        return {};
    }
    const auto chain = HostImageMipChain::generate(*image);
    EXPECT_TRUE(chain.has_value()) << chain.error().message;
    return chain ? *chain : HostImageMipChainHandle{};
}

[[nodiscard]] std::vector<TransportScalar>
constant_rgb_pixels(const std::uint32_t width, const std::uint32_t height,
                    const std::array<TransportScalar, 3U> value) {
    auto pixels = std::vector<TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(width) * height * 3U);
    for (auto index = std::uint64_t{};
         index < static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height); ++index) {
        pixels.insert(pixels.end(), value.begin(), value.end());
    }
    return pixels;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<SurfaceInteractionT<Scalar>>
planar_surface(const Vector3T<Scalar> dpdu = Vector3T<Scalar>{.x = Scalar{1}},
               const Vector3T<Scalar> dpdv = Vector3T<Scalar>{.y = Scalar{1}},
               const Normal3T<Scalar> geometric_normal = Normal3T<Scalar>{.z = Scalar{1}},
               const Normal3T<Scalar> shading_normal = Normal3T<Scalar>{.z = Scalar{1}}) {
    return SurfaceInteractionT<Scalar>::create(Point3T<Scalar>{}, geometric_normal, shading_normal,
                                               Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}},
                                               dpdu, dpdv, SurfaceIdentifiers{}, Scalar{0});
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr TextureCoordinateDifferentialsT<Scalar> full_rank_footprint() noexcept {
    return {
        .dudx = Scalar{1} / Scalar{64},
        .dvdy = Scalar{1} / Scalar{64},
    };
}

TEST(HostSurfaceMapsTest, FreezesOptionsAndYConventionAbi) {
    static_assert(std::is_standard_layout_v<HostNormalMapChannels>);
    static_assert(std::is_trivially_copyable_v<HostNormalMapChannels>);
    static_assert(std::is_standard_layout_v<HostNormalMapOptions>);
    static_assert(std::is_trivially_copyable_v<HostNormalMapOptions>);
    static_assert(std::is_standard_layout_v<HostBumpMapOptions>);
    static_assert(std::is_trivially_copyable_v<HostBumpMapOptions>);
    static_assert(std::is_standard_layout_v<ReferenceHostBumpMapOptions>);
    static_assert(std::is_trivially_copyable_v<ReferenceHostBumpMapOptions>);
    EXPECT_EQ(sizeof(HostNormalMapChannels), 12U);
    EXPECT_EQ(sizeof(HostNormalMapOptions), 32U);
    EXPECT_EQ(sizeof(HostBumpMapOptions), 24U);
    EXPECT_EQ(sizeof(ReferenceHostBumpMapOptions), 32U);
    EXPECT_TRUE(
        is_valid_tangent_space_normal_y_convention(TangentSpaceNormalYConvention::positive_v));
    EXPECT_TRUE(
        is_valid_tangent_space_normal_y_convention(TangentSpaceNormalYConvention::negative_v));
    EXPECT_FALSE(is_valid_tangent_space_normal_y_convention(
        static_cast<TangentSpaceNormalYConvention>(99U)));
}

TEST(HostSurfaceMapsTest, DecodesNormalMapWithExplicitYAndDpdvHandedness) {
    constexpr auto extent = std::uint32_t{8U};
    constexpr auto encoded = std::array<TransportScalar, 3U>{0.75F, 0.625F, 1.0F};
    const auto pixels = constant_rgb_pixels(extent, extent, encoded);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-handedness.pfm", extent, extent, 3U, pixels));
    ASSERT_TRUE(chain);
    const auto surface = planar_surface<TransportScalar>();
    const auto reference_surface = planar_surface<ReferenceScalar>();
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(reference_surface.has_value());

    const auto positive = evaluate_host_tangent_space_normal_map(
        *chain, *surface, full_rank_footprint<TransportScalar>());
    const auto reference = evaluate_host_tangent_space_normal_map(
        *chain, *reference_surface, full_rank_footprint<ReferenceScalar>());
    ASSERT_TRUE(positive.has_value()) << positive.error().message;
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    const auto inverse_length = 1.0 / std::sqrt(1.3125);
    EXPECT_NEAR(positive->x, static_cast<TransportScalar>(0.5 * inverse_length), 2.0e-6F);
    EXPECT_NEAR(positive->y, static_cast<TransportScalar>(0.25 * inverse_length), 2.0e-6F);
    EXPECT_NEAR(positive->z, static_cast<TransportScalar>(inverse_length), 2.0e-6F);
    EXPECT_NEAR(static_cast<ReferenceScalar>(positive->x), reference->x, 2.0e-6);
    EXPECT_NEAR(static_cast<ReferenceScalar>(positive->y), reference->y, 2.0e-6);
    EXPECT_NEAR(static_cast<ReferenceScalar>(positive->z), reference->z, 2.0e-6);
    EXPECT_EQ(surface->geometric_normal(), (Normal3{.z = 1.0F}));

    auto negative_options = HostNormalMapOptions{};
    negative_options.y_convention = TangentSpaceNormalYConvention::negative_v;
    const auto negative = evaluate_host_tangent_space_normal_map(
        *chain, *surface, full_rank_footprint<TransportScalar>(), negative_options);
    ASSERT_TRUE(negative.has_value()) << negative.error().message;
    EXPECT_NEAR(negative->x, positive->x, 2.0e-6F);
    EXPECT_NEAR(negative->y, -positive->y, 2.0e-6F);
    EXPECT_NEAR(negative->z, positive->z, 2.0e-6F);

    const auto mirrored_surface =
        planar_surface<TransportScalar>(Vector3{.x = 1.0F}, Vector3{.y = -1.0F});
    ASSERT_TRUE(mirrored_surface.has_value());
    const auto mirrored = evaluate_host_tangent_space_normal_map(
        *chain, *mirrored_surface, full_rank_footprint<TransportScalar>());
    ASSERT_TRUE(mirrored.has_value()) << mirrored.error().message;
    EXPECT_NEAR(mirrored->x, positive->x, 2.0e-6F);
    EXPECT_NEAR(mirrored->y, -positive->y, 2.0e-6F);
    EXPECT_NEAR(mirrored->z, positive->z, 2.0e-6F);
}

TEST(HostSurfaceMapsTest, FiltersAffineBumpAndMatchesDoubleReference) {
    constexpr auto extent = std::uint32_t{64U};
    auto ramp = std::vector<TransportScalar>{};
    ramp.reserve(static_cast<std::size_t>(extent) * extent);
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            static_cast<void>(y);
            ramp.push_back((static_cast<TransportScalar>(x) + 0.5F) /
                           static_cast<TransportScalar>(extent));
        }
    }
    auto constant = std::vector<TransportScalar>(static_cast<std::size_t>(extent) * extent, 0.4F);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto ramp_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-bump-affine.pfm", extent, extent, 1U, ramp));
    const auto constant_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-bump-constant.pfm", extent, extent, 1U, constant));
    ASSERT_TRUE(ramp_chain);
    ASSERT_TRUE(constant_chain);
    const auto surface = planar_surface<TransportScalar>();
    const auto reference_surface = planar_surface<ReferenceScalar>();
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(reference_surface.has_value());
    auto options = HostBumpMapOptions{};
    options.u_wrap = TextureWrapMode::clamp;
    options.v_wrap = TextureWrapMode::clamp;
    options.displacement_scale = 0.5F;
    auto reference_options = ReferenceHostBumpMapOptions{};
    reference_options.u_wrap = TextureWrapMode::clamp;
    reference_options.v_wrap = TextureWrapMode::clamp;
    reference_options.displacement_scale = 0.5;

    const auto bumped = evaluate_host_filtered_bump_map(
        *ramp_chain, *surface, full_rank_footprint<TransportScalar>(), options);
    const auto reference = evaluate_host_filtered_bump_map(
        *ramp_chain, *reference_surface, full_rank_footprint<ReferenceScalar>(), reference_options);
    ASSERT_TRUE(bumped.has_value()) << bumped.error().message;
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    EXPECT_NEAR(bumped->x, -1.0F / std::sqrt(5.0F), 2.0e-3F);
    EXPECT_NEAR(bumped->y, 0.0F, 2.0e-6F);
    EXPECT_NEAR(bumped->z, 2.0F / std::sqrt(5.0F), 2.0e-3F);
    EXPECT_NEAR(static_cast<ReferenceScalar>(bumped->x), reference->x, 3.0e-6);
    EXPECT_NEAR(static_cast<ReferenceScalar>(bumped->y), reference->y, 3.0e-6);
    EXPECT_NEAR(static_cast<ReferenceScalar>(bumped->z), reference->z, 3.0e-6);
    EXPECT_EQ(surface->geometric_normal(), (Normal3{.z = 1.0F}));

    const auto unchanged = evaluate_host_filtered_bump_map(
        *constant_chain, *surface, full_rank_footprint<TransportScalar>(), options);
    ASSERT_TRUE(unchanged.has_value()) << unchanged.error().message;
    EXPECT_EQ(*unchanged, surface->shading_normal());

    const auto tilted_surface = planar_surface<TransportScalar>(
        Vector3{.x = 1.0F}, Vector3{.y = 1.0F}, Normal3{.z = 1.0F}, Normal3{.x = 0.6F, .z = 0.8F});
    ASSERT_TRUE(tilted_surface.has_value()) << tilted_surface.error().message;
    const auto constant_over_tilted = evaluate_host_filtered_bump_map(
        *constant_chain, *tilted_surface, full_rank_footprint<TransportScalar>(), options);
    ASSERT_TRUE(constant_over_tilted.has_value()) << constant_over_tilted.error().message;
    EXPECT_NEAR(constant_over_tilted->x, tilted_surface->shading_normal().x, 2.0e-6F);
    EXPECT_NEAR(constant_over_tilted->y, tilted_surface->shading_normal().y, 2.0e-6F);
    EXPECT_NEAR(constant_over_tilted->z, tilted_surface->shading_normal().z, 2.0e-6F);

    options.displacement_scale = 0.0F;
    const auto zero_scale_over_tilted = evaluate_host_filtered_bump_map(
        *ramp_chain, *tilted_surface, full_rank_footprint<TransportScalar>(), options);
    ASSERT_TRUE(zero_scale_over_tilted.has_value()) << zero_scale_over_tilted.error().message;
    EXPECT_NEAR(zero_scale_over_tilted->x, tilted_surface->shading_normal().x, 2.0e-6F);
    EXPECT_NEAR(zero_scale_over_tilted->y, tilted_surface->shading_normal().y, 2.0e-6F);
    EXPECT_NEAR(zero_scale_over_tilted->z, tilted_surface->shading_normal().z, 2.0e-6F);
}

TEST(HostSurfaceMapsTest, RejectsABumpNormalAtARequiredHemisphereBoundary) {
    constexpr auto extent = std::uint32_t{64U};
    auto ramp = std::vector<TransportScalar>{};
    ramp.reserve(static_cast<std::size_t>(extent) * extent);
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            static_cast<void>(y);
            ramp.push_back((static_cast<TransportScalar>(x) + 0.5F) /
                           static_cast<TransportScalar>(extent));
        }
    }
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto chain = load_surface_map(
        *cache, write_pfm_fixture("surface-bump-shading-boundary.pfm", extent, extent, 1U, ramp));
    ASSERT_TRUE(chain);

    constexpr auto diagonal = TransportScalar{0.7071067811865475244F};
    const auto surface =
        planar_surface<TransportScalar>(Vector3{.x = 1.0F}, Vector3{.y = 1.0F}, Normal3{.z = 1.0F},
                                        Normal3{.x = diagonal, .z = diagonal});
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    auto options = HostBumpMapOptions{};
    options.u_wrap = TextureWrapMode::clamp;
    options.v_wrap = TextureWrapMode::clamp;
    // At this power-of-two scale, transport precision loses the additive dpdu.x term. The
    // normalized cross product is exactly tangent to the initial diagonal shading normal while it
    // remains above Ng; accepting it would make host and device hemisphere tests disagree.
    options.displacement_scale = std::ldexp(1.0F, 30);

    const auto boundary = evaluate_host_filtered_bump_map(
        *chain, *surface, full_rank_footprint<TransportScalar>(), options);
    ASSERT_FALSE(boundary.has_value());
    EXPECT_EQ(boundary.error().code, core::StatusCode::invalid_argument);
    // The exact zero is rounded to either side by fused arithmetic. Orientation can therefore
    // expose the same forbidden tangent boundary against Ns first or against Ng after the flip;
    // no other invalid-argument diagnostic is accepted here.
    const auto shading_boundary = boundary.error().message.find("shading") != std::string::npos;
    const auto geometric_boundary = boundary.error().message.find("geometric") != std::string::npos;
    EXPECT_TRUE(shading_boundary || geometric_boundary) << boundary.error().message;
}

TEST(HostSurfaceMapsTest, EwaMinificationSuppressesHighFrequencyBumpSlope) {
    constexpr auto extent = std::uint32_t{64U};
    auto waves = std::vector<TransportScalar>{};
    waves.reserve(static_cast<std::size_t>(extent) * extent);
    constexpr auto frequency = TransportScalar{13};
    constexpr auto phase = TransportScalar{0.3F};
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<TransportScalar>;
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            static_cast<void>(y);
            const auto u =
                (static_cast<TransportScalar>(x) + 0.5F) / static_cast<TransportScalar>(extent);
            waves.push_back(0.5F + 0.45F * std::sin(two_pi * frequency * u + phase));
        }
    }
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto chain = load_surface_map(
        *cache, write_pfm_fixture("surface-bump-minification.pfm", extent, extent, 1U, waves));
    ASSERT_TRUE(chain);
    const auto surface = planar_surface<TransportScalar>();
    ASSERT_TRUE(surface.has_value());
    auto options = HostBumpMapOptions{};
    options.displacement_scale = 0.02F;
    constexpr auto fine = TextureCoordinateDifferentials{
        .dudx = 1.0F / 1024.0F,
        .dvdy = 1.0F / 1024.0F,
    };
    constexpr auto minified = TextureCoordinateDifferentials{
        .dudx = 0.18F,
        .dvdy = 0.16F,
    };
    const auto fine_normal = evaluate_host_filtered_bump_map(*chain, *surface, fine, options);
    const auto minified_normal =
        evaluate_host_filtered_bump_map(*chain, *surface, minified, options);
    ASSERT_TRUE(fine_normal.has_value()) << fine_normal.error().message;
    ASSERT_TRUE(minified_normal.has_value()) << minified_normal.error().message;
    EXPECT_LT(fine_normal->z, 0.97F);
    EXPECT_GT(minified_normal->z, 0.995F);
    EXPECT_GT(minified_normal->z, fine_normal->z + 0.03F);
}

TEST(HostSurfaceMapsTest, RejectsColorDataInvalidChannelsAndInvalidOptions) {
    constexpr auto extent = std::uint32_t{8U};
    const auto pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{0.5F, 0.5F, 1.0F});
    const auto path = write_pfm_fixture("surface-normal-contract.pfm", extent, extent, 3U, pixels);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto data_chain = load_surface_map(*cache, path);
    const auto color_chain = load_surface_map(*cache, path, TextureColorSpace::srgb);
    ASSERT_TRUE(data_chain);
    ASSERT_TRUE(color_chain);
    const auto surface = planar_surface<TransportScalar>();
    ASSERT_TRUE(surface.has_value());
    constexpr auto footprint = full_rank_footprint<TransportScalar>();

    const auto color = evaluate_host_tangent_space_normal_map(*color_chain, *surface, footprint);
    ASSERT_FALSE(color.has_value());
    EXPECT_EQ(color.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(color.error().message.find("data"), std::string::npos);

    auto duplicate = HostNormalMapOptions{};
    duplicate.channels.y = duplicate.channels.x;
    const auto duplicate_result =
        evaluate_host_tangent_space_normal_map(*data_chain, *surface, footprint, duplicate);
    ASSERT_FALSE(duplicate_result.has_value());
    EXPECT_NE(duplicate_result.error().message.find("distinct"), std::string::npos);

    auto absent = HostNormalMapOptions{};
    absent.channels.z = 3U;
    const auto absent_result =
        evaluate_host_tangent_space_normal_map(*data_chain, *surface, footprint, absent);
    ASSERT_FALSE(absent_result.has_value());
    EXPECT_NE(absent_result.error().message.find("exist"), std::string::npos);

    auto invalid_y = HostNormalMapOptions{};
    invalid_y.y_convention = static_cast<TangentSpaceNormalYConvention>(99U);
    EXPECT_FALSE(
        evaluate_host_tangent_space_normal_map(*data_chain, *surface, footprint, invalid_y));

    auto invalid_wrap = HostNormalMapOptions{};
    invalid_wrap.u_wrap = static_cast<TextureWrapMode>(99U);
    EXPECT_FALSE(
        evaluate_host_tangent_space_normal_map(*data_chain, *surface, footprint, invalid_wrap));

    auto invalid_anisotropy = HostNormalMapOptions{};
    invalid_anisotropy.ewa_limits.maximum_anisotropy = 0U;
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*data_chain, *surface, footprint,
                                                        invalid_anisotropy));

    auto bump_absent = HostBumpMapOptions{};
    bump_absent.height_channel = 3U;
    EXPECT_FALSE(evaluate_host_filtered_bump_map(*data_chain, *surface, footprint, bump_absent));
    auto bump_non_finite = HostBumpMapOptions{};
    bump_non_finite.displacement_scale = std::numeric_limits<TransportScalar>::quiet_NaN();
    EXPECT_FALSE(
        evaluate_host_filtered_bump_map(*data_chain, *surface, footprint, bump_non_finite));
}

TEST(HostSurfaceMapsTest, RejectsDegenerateFootprintsTangentsAndEncodedNormals) {
    constexpr auto extent = std::uint32_t{8U};
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto valid_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{0.5F, 0.5F, 1.0F});
    const auto zero_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{0.5F, 0.5F, 0.5F});
    const auto lower_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{0.5F, 0.5F, 0.0F});
    const auto range_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{1.25F, 0.5F, 1.0F});
    const auto steep_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{1.0F, 0.5F, 0.51F});
    const auto valid_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-valid.pfm", extent, extent, 3U, valid_pixels));
    const auto zero_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-zero.pfm", extent, extent, 3U, zero_pixels));
    const auto lower_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-lower.pfm", extent, extent, 3U, lower_pixels));
    const auto range_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-range.pfm", extent, extent, 3U, range_pixels));
    const auto steep_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-steep.pfm", extent, extent, 3U, steep_pixels));
    ASSERT_TRUE(valid_chain);
    ASSERT_TRUE(zero_chain);
    ASSERT_TRUE(lower_chain);
    ASSERT_TRUE(range_chain);
    ASSERT_TRUE(steep_chain);
    const auto surface = planar_surface<TransportScalar>();
    ASSERT_TRUE(surface.has_value());
    constexpr auto valid_footprint = full_rank_footprint<TransportScalar>();

    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *surface,
                                                        TextureCoordinateDifferentials{}));
    constexpr auto rank_one = TextureCoordinateDifferentials{
        .dudx = 1.0F,
        .dvdx = 2.0F,
        .dudy = 2.0F,
        .dvdy = 4.0F,
    };
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *surface, rank_one));
    auto non_finite = valid_footprint;
    non_finite.dudx = std::numeric_limits<TransportScalar>::infinity();
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *surface, non_finite));

    const auto zero_dpdu = planar_surface<TransportScalar>(Vector3{}, Vector3{.y = 1.0F});
    const auto zero_dpdv = planar_surface<TransportScalar>(Vector3{.x = 1.0F}, Vector3{});
    const auto ambiguous = planar_surface<TransportScalar>(Vector3{.x = 1.0F}, Vector3{.x = 1.0F});
    ASSERT_TRUE(zero_dpdu.has_value());
    ASSERT_TRUE(zero_dpdv.has_value());
    ASSERT_TRUE(ambiguous.has_value());
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *zero_dpdu, valid_footprint));
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *zero_dpdv, valid_footprint));
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*valid_chain, *ambiguous, valid_footprint));
    EXPECT_FALSE(evaluate_host_filtered_bump_map(*valid_chain, *zero_dpdu, valid_footprint));

    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*zero_chain, *surface, valid_footprint));
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*lower_chain, *surface, valid_footprint));
    EXPECT_FALSE(evaluate_host_tangent_space_normal_map(*range_chain, *surface, valid_footprint));

    constexpr auto tilted_ns = Normal3{.x = 0.8F, .z = 0.6F};
    const auto tilted_surface = planar_surface<TransportScalar>(
        Vector3{.x = 1.0F}, Vector3{.y = 1.0F}, Normal3{.z = 1.0F}, tilted_ns);
    ASSERT_TRUE(tilted_surface.has_value());
    const auto below_geometry =
        evaluate_host_tangent_space_normal_map(*steep_chain, *tilted_surface, valid_footprint);
    ASSERT_FALSE(below_geometry.has_value());
    EXPECT_NE(below_geometry.error().message.find("geometric"), std::string::npos);
}

TEST(HostSurfaceMapsTest, PropagatesEwaBudgetsAndDoesNotBypassZeroBumpScale) {
    constexpr auto extent = std::uint32_t{64U};
    const auto normal_pixels =
        constant_rgb_pixels(extent, extent, std::array<TransportScalar, 3U>{0.5F, 0.5F, 1.0F});
    auto height_pixels =
        std::vector<TransportScalar>(static_cast<std::size_t>(extent) * extent, 0.5F);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto normal_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-normal-budget.pfm", extent, extent, 3U, normal_pixels));
    const auto bump_chain = load_surface_map(
        *cache, write_pfm_fixture("surface-bump-budget.pfm", extent, extent, 1U, height_pixels));
    ASSERT_TRUE(normal_chain);
    ASSERT_TRUE(bump_chain);
    const auto surface = planar_surface<TransportScalar>();
    ASSERT_TRUE(surface.has_value());
    constexpr auto broad = TextureCoordinateDifferentials{
        .dudx = 0.2F,
        .dvdy = 0.2F,
    };
    auto normal_options = HostNormalMapOptions{};
    normal_options.ewa_limits.maximum_texel_visits = 1U;
    const auto normal =
        evaluate_host_tangent_space_normal_map(*normal_chain, *surface, broad, normal_options);
    ASSERT_FALSE(normal.has_value());
    EXPECT_EQ(normal.error().code, core::StatusCode::resource_exhausted);
    EXPECT_EQ(normal.error().message, "The EWA texture footprint exceeds its texel-visit budget.");

    auto bump_options = HostBumpMapOptions{};
    bump_options.displacement_scale = 0.0F;
    bump_options.ewa_limits.maximum_texel_visits = 1U;
    const auto bump = evaluate_host_filtered_bump_map(*bump_chain, *surface, broad, bump_options);
    ASSERT_FALSE(bump.has_value());
    EXPECT_EQ(bump.error().code, core::StatusCode::resource_exhausted);
    EXPECT_EQ(bump.error().message, "The EWA texture footprint exceeds its texel-visit budget.");
}

[[nodiscard]] std::filesystem::path checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return {};
    }
    const auto output = value_size > 1U ? std::filesystem::path{value} : std::filesystem::path{};
    std::free(value);
    return output;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    return value == nullptr || *value == '\0' ? std::filesystem::path{}
                                              : std::filesystem::path{value};
#endif
}

#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
[[nodiscard]] std::vector<TransportScalar> preview_normal_pixels(const std::uint32_t extent) {
    auto pixels = std::vector<TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(extent) * extent * 3U);
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<TransportScalar>;
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            const auto u =
                (static_cast<TransportScalar>(x) + 0.5F) / static_cast<TransportScalar>(extent);
            const auto v =
                (static_cast<TransportScalar>(y) + 0.5F) / static_cast<TransportScalar>(extent);
            const auto nx = 0.42F * std::sin(two_pi * 4.0F * u) * std::sin(two_pi * 2.0F * v);
            const auto ny = 0.30F * std::cos(two_pi * 3.0F * u + two_pi * v);
            const auto nz = std::sqrt(std::max(0.05F, 1.0F - nx * nx - ny * ny));
            pixels.push_back(0.5F * (nx + 1.0F));
            pixels.push_back(0.5F * (ny + 1.0F));
            pixels.push_back(0.5F * (nz + 1.0F));
        }
    }
    return pixels;
}

[[nodiscard]] std::vector<TransportScalar> preview_bump_pixels(const std::uint32_t extent) {
    auto pixels = std::vector<TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(extent) * extent);
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<TransportScalar>;
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            const auto u =
                (static_cast<TransportScalar>(x) + 0.5F) / static_cast<TransportScalar>(extent);
            const auto v =
                (static_cast<TransportScalar>(y) + 0.5F) / static_cast<TransportScalar>(extent);
            pixels.push_back(0.5F +
                             0.22F * std::sin(two_pi * 10.0F * u) * std::sin(two_pi * 6.0F * v));
        }
    }
    return pixels;
}

[[nodiscard]] core::Result<SurfaceInteraction>
preview_sphere_surface(const TransportScalar x, const TransportScalar y, const TransportScalar z) {
    const auto normal = Normal3{.x = x, .y = y, .z = z};
    constexpr auto pi = std::numbers::pi_v<TransportScalar>;
    constexpr auto two_pi = 2.0F * pi;
    auto phi = std::atan2(z, x);
    if (phi < 0.0F) {
        phi += two_pi;
    }
    const auto theta = std::atan2(std::hypot(x, z), y);
    const auto sin_theta = std::sin(theta);
    const auto cos_theta = std::cos(theta);
    const auto sin_phi = std::sin(phi);
    const auto cos_phi = std::cos(phi);
    return SurfaceInteraction::create(
        Point3{.x = x, .y = y, .z = z}, normal, normal, Point2{.x = phi / two_pi, .y = theta / pi},
        Vector3{.x = -two_pi * sin_phi * sin_theta, .y = 0.0F, .z = two_pi * cos_phi * sin_theta},
        Vector3{.x = pi * cos_phi * cos_theta, .y = -pi * sin_theta, .z = pi * sin_phi * cos_theta},
        SurfaceIdentifiers{}, 0.0F);
}

[[nodiscard]] core::Status write_surface_map_preview(const HostImageMipChain& normal_chain,
                                                     const HostImageMipChain& bump_chain,
                                                     const std::filesystem::path& output) {
    constexpr auto extent = std::uint32_t{800U};
    constexpr auto sphere_radius = TransportScalar{175};
    constexpr auto center_y = TransportScalar{400};
    constexpr auto centers = std::array<TransportScalar, 2U>{205.0F, 595.0F};
    constexpr auto footprint = TextureCoordinateDifferentials{
        .dudx = 1.0F / 256.0F,
        .dvdy = 1.0F / 256.0F,
    };
    auto bump_options = HostBumpMapOptions{};
    bump_options.displacement_scale = 0.075F;
    auto film = Film::create(RenderExtent{.width = extent, .height = extent});
    if (!film) {
        return std::unexpected(film.error());
    }
    constexpr auto light = Vector3{.x = -0.35599533F, .y = 0.55942124F, .z = 0.74867600F};
    for (auto py = std::uint32_t{}; py < extent; ++py) {
        for (auto px = std::uint32_t{}; px < extent; ++px) {
            const auto raster_x = static_cast<TransportScalar>(px) + 0.5F;
            const auto raster_y = static_cast<TransportScalar>(py) + 0.5F;
            auto color = LinearRGB{
                .red = 0.018F + 0.025F * (raster_y / static_cast<TransportScalar>(extent)),
                .green = 0.022F + 0.030F * (raster_y / static_cast<TransportScalar>(extent)),
                .blue = 0.032F + 0.045F * (raster_y / static_cast<TransportScalar>(extent)),
            };
            for (auto sphere_index = std::size_t{}; sphere_index < centers.size(); ++sphere_index) {
                const auto x = (raster_x - centers[sphere_index]) / sphere_radius;
                const auto y = (center_y - raster_y) / sphere_radius;
                const auto radius_squared = std::fma(x, x, y * y);
                if (radius_squared >= 1.0F) {
                    continue;
                }
                const auto z = std::sqrt(1.0F - radius_squared);
                const auto surface = preview_sphere_surface(x, y, z);
                if (!surface) {
                    return std::unexpected(surface.error());
                }
                auto normal = core::Result<Normal3>{};
                if (sphere_index == 0U) {
                    normal =
                        evaluate_host_tangent_space_normal_map(normal_chain, *surface, footprint);
                } else {
                    normal = evaluate_host_filtered_bump_map(bump_chain, *surface, footprint,
                                                             bump_options);
                }
                if (!normal) {
                    return std::unexpected(normal.error());
                }
                const auto diffuse = std::max(0.0F, dot(*normal, light));
                const auto facing = std::max(0.0F, normal->z);
                const auto intensity = 0.10F + 0.78F * diffuse + 0.12F * facing;
                if (sphere_index == 0U) {
                    color = LinearRGB{.red = 0.82F * intensity,
                                      .green = 0.44F * intensity,
                                      .blue = 0.22F * intensity};
                } else {
                    color = LinearRGB{.red = 0.22F * intensity,
                                      .green = 0.52F * intensity,
                                      .blue = 0.86F * intensity};
                }
                break;
            }
            const auto status = film->add_sample(px, py, color, 1.0F);
            if (!status) {
                return status;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(HostSurfaceMapsTest, WritesStableNormalAndBumpSpherePreview) {
    const auto output = checksum_output_path();
    if (output.empty()) {
        GTEST_SKIP() << "The explicit PNG checksum output path was not supplied.";
    }
#if !defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
    FAIL() << "The normal and bump sphere preview requires the explicit PNG capability.";
#else
    constexpr auto texture_extent = std::uint32_t{64U};
    const auto normal_pixels = preview_normal_pixels(texture_extent);
    const auto bump_pixels = preview_bump_pixels(texture_extent);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto normal_chain =
        load_surface_map(*cache, write_pfm_fixture("surface-map-preview-normal.pfm", texture_extent,
                                                   texture_extent, 3U, normal_pixels));
    const auto bump_chain =
        load_surface_map(*cache, write_pfm_fixture("surface-map-preview-bump.pfm", texture_extent,
                                                   texture_extent, 1U, bump_pixels));
    ASSERT_TRUE(normal_chain);
    ASSERT_TRUE(bump_chain);
    const auto status = write_surface_map_preview(*normal_chain, *bump_chain, output);
    ASSERT_TRUE(status.has_value()) << status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output));
    testing::Test::RecordProperty(
        "preview_layout",
        "left=tangent-space-normal-EWA-sphere;right=filtered-bump-EWA-sphere;800x800");
#endif
}

} // namespace
} // namespace blackframe::renderer
