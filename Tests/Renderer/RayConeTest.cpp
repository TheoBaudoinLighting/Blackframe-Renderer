#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/RayCone.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {
namespace {

struct FootprintComparison final {
    Point3 target_position;
    float differential_radius;
    float cone_radius;
    bool reflection;
};

[[nodiscard]] Ray make_ray(const Point3 origin, const Vector3 direction) {
    return Ray::create(origin, direction, 0.0F, std::numeric_limits<float>::infinity(), 0.0F,
                       AllRayVisibility, VacuumMedium)
        .value();
}

[[nodiscard]] core::Result<float> plane_parameter(const Ray& ray, const float plane_z) {
    if (!std::isfinite(ray.direction().z) || ray.direction().z == 0.0F) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The ray-cone comparison ray is parallel to its target plane.",
        });
    }
    const auto parameter = (plane_z - ray.origin().z) / ray.direction().z;
    if (!std::isfinite(parameter) || !(parameter >= 0.0F)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The ray-cone comparison plane is not in front of the ray.",
        });
    }
    return parameter;
}

[[nodiscard]] SurfaceInteraction make_planar_interaction(const Point3 position,
                                                         const Normal3 normal) {
    return SurfaceInteraction::create(position, normal, normal, Point2{}, Vector3{.x = 1.0F},
                                      Vector3{.y = 1.0F}, SurfaceIdentifiers{}, 0.0F)
        .value();
}

[[nodiscard]] core::Result<FootprintComparison>
compare_specular_footprints(const PinholeCamera& camera, const Point2 raster_sample,
                            const bool reflection) {
    const auto camera_differential = camera.generate_primary_ray_differential(raster_sample, 0.0F);
    const auto camera_cone = camera.generate_primary_ray_cone(raster_sample, 0.0F);
    if (!camera_differential) {
        return std::unexpected(camera_differential.error());
    }
    if (!camera_cone) {
        return std::unexpected(camera_cone.error());
    }

    constexpr auto interface_normal = Normal3{.z = 1.0F};
    const auto interface_parameter = plane_parameter(camera_differential->ray(), 0.0F);
    if (!interface_parameter) {
        return std::unexpected(interface_parameter.error());
    }
    const auto interface_position = camera_differential->ray().at(*interface_parameter);
    if (!interface_position) {
        return std::unexpected(interface_position.error());
    }
    const auto interface = make_planar_interaction(*interface_position, interface_normal);
    const auto interface_differentials =
        surface_point_differentials(*camera_differential, interface);
    const auto interface_cone =
        advance_ray_cone(*camera_cone, camera_differential->ray(), *interface_parameter);
    if (!interface_differentials) {
        return std::unexpected(interface_differentials.error());
    }
    if (!interface_cone) {
        return std::unexpected(interface_cone.error());
    }

    const auto rx_position = *interface_position + interface_differentials->dpdx;
    const auto ry_position = *interface_position + interface_differentials->dpdy;
    auto next_direction = Vector3{};
    auto target_z = 0.0F;
    auto propagated_differential = std::optional<RayDifferential>{};
    auto closure_set = ClosureSet{};
    auto event = ScatteringLobe::none;

    if (reflection) {
        const auto incident = camera_differential->ray().direction();
        next_direction = incident - 2.0F * dot(incident, interface_normal) * Vector3{.z = 1.0F};
        target_z = 4.0F;
        const auto next_ray = make_ray(*interface_position, next_direction);
        const auto reflected = propagate_specular_reflection(
            *camera_differential, next_ray, interface_normal, rx_position, interface_normal,
            ry_position, interface_normal);
        if (!reflected) {
            return std::unexpected(reflected.error());
        }
        propagated_differential = *reflected;
        if (closure_set.append_specular_reflection(TransportSpectrum{1.0F}) !=
            ClosureAppendStatus::appended) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The mirror comparison closure could not be constructed.",
            });
        }
        event = ScatteringLobe::specular | ScatteringLobe::reflection;
    } else {
        constexpr auto relative_eta = 1.0F / 1.5F;
        const auto incident = camera_differential->ray().direction();
        const auto tangent =
            Vector3{.x = relative_eta * incident.x, .y = relative_eta * incident.y};
        const auto tangent_length = std::hypot(tangent.x, tangent.y);
        next_direction = Vector3{
            .x = tangent.x,
            .y = tangent.y,
            .z = -std::sqrt((1.0F - tangent_length) * (1.0F + tangent_length)),
        };
        target_z = -4.0F;
        const auto next_ray = make_ray(*interface_position, next_direction);
        const auto transmitted = propagate_specular_transmission(
            *camera_differential, next_ray, interface_normal, rx_position, interface_normal,
            ry_position, interface_normal, true, 1.0F, 1.5F);
        if (!transmitted) {
            return std::unexpected(transmitted.error());
        }
        if (!*transmitted) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The entering ray-cone comparison unexpectedly reached TIR.",
            });
        }
        propagated_differential = **transmitted;
        if (closure_set.append_specular_transmission(TransportSpectrum{1.0F}, 1.0F, 1.5F) !=
            ClosureAppendStatus::appended) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The transmission comparison closure could not be constructed.",
            });
        }
        event = ScatteringLobe::specular | ScatteringLobe::transmission;
    }

    const auto outgoing_local = -camera_differential->ray().direction();
    const auto propagated_cone = propagate_ray_cone_scattering(
        *interface_cone, closure_set.closures().front(), event, outgoing_local, next_direction);
    if (!propagated_cone) {
        return std::unexpected(propagated_cone.error());
    }
    const auto next_ray = make_ray(*interface_position, next_direction);
    const auto target_parameter = plane_parameter(next_ray, target_z);
    if (!target_parameter) {
        return std::unexpected(target_parameter.error());
    }
    const auto target_position = next_ray.at(*target_parameter);
    if (!target_position) {
        return std::unexpected(target_position.error());
    }
    const auto target = make_planar_interaction(*target_position, interface_normal);
    const auto target_differentials = surface_point_differentials(*propagated_differential, target);
    if (!target_differentials) {
        return std::unexpected(target_differentials.error());
    }
    const auto differential_radius =
        ray_differential_surface_footprint_radius(*target_differentials);
    const auto cone_radius = ray_cone_surface_footprint_radius(*propagated_cone, next_ray,
                                                               *target_parameter, interface_normal);
    if (!differential_radius) {
        return std::unexpected(differential_radius.error());
    }
    if (!cone_radius) {
        return std::unexpected(cone_radius.error());
    }
    return FootprintComparison{
        .target_position = *target_position,
        .differential_radius = *differential_radius,
        .cone_radius = *cone_radius,
        .reflection = reflection,
    };
}

[[nodiscard]] float filtered_pattern(const Point3 position, const float radius) {
    constexpr auto angular_frequency = 14.0F;
    const auto attenuation =
        std::exp(-0.5F * angular_frequency * angular_frequency * radius * radius);
    return 0.5F + 0.45F * attenuation * std::sin(angular_frequency * position.x) *
                      std::sin(angular_frequency * position.y);
}

[[nodiscard]] LinearRGB comparison_color(const FootprintComparison comparison,
                                         const bool use_cone) {
    const auto value =
        filtered_pattern(comparison.target_position,
                         use_cone ? comparison.cone_radius : comparison.differential_radius);
    return comparison.reflection
               ? LinearRGB{.red = 0.96F * value, .green = 0.72F * value, .blue = 0.32F * value}
               : LinearRGB{.red = 0.32F * value, .green = 0.72F * value, .blue = 0.96F * value};
}

[[nodiscard]] std::optional<std::filesystem::path> checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1U ? std::optional{std::filesystem::path{value}} : std::nullopt;
    std::free(value);
    return path;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path{value};
#endif
}

TEST(RayConeTest, FreezesRepresentationAndRejectsInvalidFootprints) {
    static_assert(std::is_standard_layout_v<RayCone>);
    static_assert(std::is_trivially_copyable_v<RayCone>);
    static_assert(!std::is_default_constructible_v<RayCone>);
    static_assert(sizeof(RayCone) == 2U * sizeof(float));

    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto infinity = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(RayCone::create(-1.0F, 0.0F).has_value());
    EXPECT_FALSE(RayCone::create(0.0F, -1.0F).has_value());
    EXPECT_FALSE(RayCone::create(nan, 0.0F).has_value());
    EXPECT_FALSE(RayCone::create(0.0F, infinity).has_value());

    const auto cone = RayCone::create(0.25F, 0.1F).value();
    EXPECT_FALSE(advance_ray_cone(cone, -1.0F).has_value());
    EXPECT_FALSE(ray_cone_surface_footprint_radius(cone, Vector3{.z = -1.0F}, Normal3{.x = 1.0F})
                     .has_value());
}

TEST(RayConeTest, DerivesThePinholeConeAndUsesWorldSpaceDistance) {
    constexpr auto extent = RenderExtent{.width = 800U, .height = 800U};
    const auto frame =
        OrthonormalFrame::from_normal_and_tangent(Normal3{.z = 1.0F}, Vector3{.x = 1.0F});
    ASSERT_TRUE(frame.has_value());
    const auto camera = PinholeCamera::create(
        Point3{}, *frame, extent, std::numbers::pi_v<float> / 3.0F, 0.0F,
        std::numeric_limits<float>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());
    const auto cone = camera->generate_primary_ray_cone(Point2{.x = 400.0F, .y = 400.0F}, 0.0F);
    ASSERT_TRUE(cone.has_value()) << cone.error().message;
    EXPECT_EQ(cone->width(), 0.0F);
    EXPECT_NEAR(cone->spread(), 2.0F * std::tan(std::numbers::pi_v<float> / 6.0F) / 800.0F,
                3.0e-6F);

    const auto explicit_cone = RayCone::create(0.25F, 0.1F).value();
    const auto non_unit_ray = make_ray(Point3{}, Vector3{.z = -2.0F});
    const auto advanced = advance_ray_cone(explicit_cone, non_unit_ray, 3.0F);
    ASSERT_TRUE(advanced.has_value()) << advanced.error().message;
    EXPECT_FLOAT_EQ(advanced->width(), 0.85F);
    EXPECT_FLOAT_EQ(advanced->spread(), 0.1F);

    const auto smallest = std::numeric_limits<float>::denorm_min();
    const auto subnormal_ray = make_ray(Point3{}, Vector3{.z = smallest});
    EXPECT_FALSE(advance_ray_cone(explicit_cone, subnormal_ray, smallest).has_value());
}

TEST(RayConeTest, ConvertsDecenteredAnisotropicDifferentialsConservatively) {
    const auto central = make_ray(Point3{}, Vector3{.z = -1.0F});
    const auto differential =
        RayDifferential::create(central, Point3{.x = 0.2F}, Vector3{.x = 0.1F, .z = -1.0F},
                                Point3{.y = 0.3F}, Vector3{.y = 0.2F, .z = -1.0F});
    ASSERT_TRUE(differential.has_value()) << differential.error().message;
    const auto cone = ray_cone_from_differential(*differential);
    ASSERT_TRUE(cone.has_value()) << cone.error().message;
    EXPECT_NEAR(cone->width(), 0.3F, 2.0e-6F);
    EXPECT_GT(cone->spread(), 0.19F);
    EXPECT_LT(cone->spread(), 0.21F);
}

TEST(RayConeTest, RemainsCloseToDifferentialsAcrossIdealPlanarPaths) {
    constexpr auto extent = RenderExtent{.width = 800U, .height = 800U};
    const auto frame =
        OrthonormalFrame::from_normal_and_tangent(Normal3{.z = 1.0F}, Vector3{.x = 1.0F});
    ASSERT_TRUE(frame.has_value());
    const auto camera = PinholeCamera::create(
        Point3{.z = 2.0F}, *frame, extent, std::numbers::pi_v<float> / 3.0F, 0.0F,
        std::numeric_limits<float>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    for (const auto reflection : {false, true}) {
        for (const auto raster :
             {Point2{.x = 400.5F, .y = 400.5F}, Point2{.x = 330.5F, .y = 470.5F},
              Point2{.x = 470.5F, .y = 330.5F}}) {
            const auto comparison = compare_specular_footprints(*camera, raster, reflection);
            ASSERT_TRUE(comparison.has_value()) << comparison.error().message;
            ASSERT_GT(comparison->differential_radius, 0.0F);
            EXPECT_NEAR(comparison->cone_radius, comparison->differential_radius,
                        0.035F * comparison->differential_radius);
        }
    }

    const auto grazing_camera = PinholeCamera::create(
        Point3{.z = 2.0F}, *frame, extent, 140.0F * std::numbers::pi_v<float> / 180.0F, 0.0F,
        std::numeric_limits<float>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(grazing_camera.has_value());
    const auto grazing =
        compare_specular_footprints(*grazing_camera, Point2{.x = 720.5F, .y = 400.5F}, false);
    ASSERT_TRUE(grazing.has_value()) << grazing.error().message;
    EXPECT_GE(grazing->cone_radius, grazing->differential_radius);
    EXPECT_LE(grazing->cone_radius, 2.05F * grazing->differential_radius);
}

TEST(RayConeTest, AppliesExplicitLobeSpreadWithoutAHiddenDefault) {
    auto closures = ClosureSet{};
    ASSERT_EQ(closures.append_lambertian_reflection(TransportSpectrum{1.0F}),
              ClosureAppendStatus::appended);
    const auto cone = RayCone::create(0.01F, 0.02F).value();
    const auto diffuse = propagate_ray_cone_scattering(
        cone, closures.closures().front(), ScatteringLobe::diffuse | ScatteringLobe::reflection,
        Vector3{.z = 1.0F}, Vector3{.z = 1.0F});
    ASSERT_TRUE(diffuse.has_value()) << diffuse.error().message;
    EXPECT_FLOAT_EQ(diffuse->spread(), 1.0F);

    const auto mismatch = propagate_ray_cone_scattering(
        cone, closures.closures().front(), ScatteringLobe::specular | ScatteringLobe::transmission,
        Vector3{.z = 1.0F}, Vector3{.z = -1.0F});
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error().code, core::StatusCode::invalid_argument);

    auto mirror = ClosureSet{};
    ASSERT_EQ(mirror.append_specular_reflection(TransportSpectrum{1.0F}),
              ClosureAppendStatus::appended);
    const auto wrong_hemisphere = propagate_ray_cone_scattering(
        cone, mirror.closures().front(), ScatteringLobe::specular | ScatteringLobe::reflection,
        Vector3{.z = 1.0F}, Vector3{.z = -1.0F});
    ASSERT_FALSE(wrong_hemisphere.has_value());
    EXPECT_EQ(wrong_hemisphere.error().code, core::StatusCode::invalid_argument);

    auto unknown = mirror.closures().front();
    unknown.kind = static_cast<ClosureKind>(std::numeric_limits<std::uint32_t>::max());
    EXPECT_FALSE(propagate_ray_cone_scattering(
                     cone, unknown, ScatteringLobe::specular | ScatteringLobe::reflection,
                     Vector3{.z = 1.0F}, Vector3{.z = 1.0F})
                     .has_value());

    auto invalid_roughness = Closure{};
    invalid_roughness.kind = ClosureKind::rough_conductor_reflection;
    invalid_roughness.lobes = ScatteringLobe::glossy | ScatteringLobe::reflection;
    invalid_roughness.parameters[8U] = -0.1F;
    invalid_roughness.parameters[9U] = -0.2F;
    EXPECT_FALSE(propagate_ray_cone_scattering(cone, invalid_roughness,
                                               ScatteringLobe::glossy | ScatteringLobe::reflection,
                                               Vector3{.z = 1.0F}, Vector3{.z = 1.0F})
                     .has_value());

    auto conductor = ClosureSet{};
    const auto unit_spectrum = TransportSpectrum{.values = {1.0F, 1.0F, 1.0F, 1.0F}};
    ASSERT_EQ(conductor.append_rough_conductor_reflection(unit_spectrum, unit_spectrum,
                                                          unit_spectrum, 0.1F, 0.4F),
              ClosureAppendStatus::appended);
    const auto anisotropic = propagate_ray_cone_scattering(
        cone, conductor.closures().front(), ScatteringLobe::glossy | ScatteringLobe::reflection,
        Vector3{.z = 1.0F}, Vector3{.z = 1.0F});
    ASSERT_TRUE(anisotropic.has_value()) << anisotropic.error().message;
    EXPECT_FLOAT_EQ(anisotropic->spread(), 0.4F);
}

TEST(RayConeTest, BoundsBothSnellDerivativesAtGrazingIncidence) {
    auto glass = ClosureSet{};
    ASSERT_EQ(glass.append_specular_transmission(TransportSpectrum{1.0F}, 1.0F, 1.5F),
              ClosureAppendStatus::appended);
    const auto cone = RayCone::create(0.01F, 0.02F).value();
    constexpr auto incident_cosine = 0.17364818F;
    constexpr auto incident_sine = 0.98480775F;
    constexpr auto eta_ratio = 1.0F / 1.5F;
    const auto transmitted_sine = eta_ratio * incident_sine;
    const auto transmitted_cosine =
        std::sqrt((1.0F - transmitted_sine) * (1.0F + transmitted_sine));
    const auto transmitted = propagate_ray_cone_scattering(
        cone, glass.closures().front(), ScatteringLobe::specular | ScatteringLobe::transmission,
        Vector3{.x = incident_sine, .z = incident_cosine},
        Vector3{.x = -transmitted_sine, .z = -transmitted_cosine});
    ASSERT_TRUE(transmitted.has_value()) << transmitted.error().message;

    const auto azimuthal_spread = cone.spread() * eta_ratio;
    const auto meridional_spread = cone.spread() * eta_ratio * incident_cosine / transmitted_cosine;
    EXPECT_NEAR(transmitted->spread(), std::max(azimuthal_spread, meridional_spread), 2.0e-7F);
    EXPECT_NEAR(transmitted->width(),
                cone.width() * std::max(1.0F, transmitted_cosine / incident_cosine), 2.0e-7F);
    EXPECT_GT(transmitted->spread(), meridional_spread * 4.0F);
}

TEST(RayConeTest, WritesStableDifferentialComparisonPreview) {
    const auto output = checksum_output_path();
    if (!output) {
        GTEST_SKIP() << "The explicit PNG checksum output path was not supplied.";
    }
#if BLACKFRAME_HAS_PNG_PREVIEW
    constexpr auto comparison_extent = RenderExtent{.width = 800U, .height = 399U};
    constexpr auto preview_extent = RenderExtent{.width = 800U, .height = 800U};
    const auto frame =
        OrthonormalFrame::from_normal_and_tangent(Normal3{.z = 1.0F}, Vector3{.x = 1.0F});
    ASSERT_TRUE(frame.has_value());
    const auto camera = PinholeCamera::create(
        Point3{.z = 2.0F}, *frame, comparison_extent, std::numbers::pi_v<float> / 3.0F, 0.0F,
        std::numeric_limits<float>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());
    auto differential_film = Film::create(comparison_extent);
    auto cone_film = Film::create(comparison_extent);
    auto preview = Film::create(preview_extent);
    ASSERT_TRUE(differential_film.has_value());
    ASSERT_TRUE(cone_film.has_value());
    ASSERT_TRUE(preview.has_value());

    for (auto y = std::uint32_t{}; y < comparison_extent.height; ++y) {
        for (auto x = std::uint32_t{}; x < comparison_extent.width; ++x) {
            const auto comparison = compare_specular_footprints(
                *camera,
                Point2{.x = static_cast<float>(x) + 0.5F, .y = static_cast<float>(y) + 0.5F},
                x < comparison_extent.width / 2U);
            ASSERT_TRUE(comparison.has_value()) << comparison.error().message;
            const auto differential_color = comparison_color(*comparison, false);
            const auto cone_color = comparison_color(*comparison, true);
            ASSERT_TRUE(differential_film->add_sample(x, y, differential_color, 1.0F).has_value());
            ASSERT_TRUE(cone_film->add_sample(x, y, cone_color, 1.0F).has_value());
            ASSERT_TRUE(preview->add_sample(x, y, differential_color, 1.0F).has_value());
            ASSERT_TRUE(preview->add_sample(x, y + 401U, cone_color, 1.0F).has_value());
        }
    }
    for (auto y = 399U; y <= 400U; ++y) {
        for (auto x = std::uint32_t{}; x < preview_extent.width; ++x) {
            ASSERT_TRUE(
                preview
                    ->add_sample(x, y, LinearRGB{.red = 0.96F, .green = 0.96F, .blue = 0.96F}, 1.0F)
                    .has_value());
        }
    }

    const auto metrics = compute_linear_metrics(*cone_film, *differential_film);
    const auto psnr = compute_display_psnr(*cone_film, *differential_film);
    ASSERT_TRUE(metrics.has_value()) << metrics.error().message;
    ASSERT_TRUE(psnr.has_value()) << psnr.error().message;
    EXPECT_LE(metrics->mse, 1.0e-5);
    EXPECT_GE(psnr->psnr, 50.0);
    testing::Test::RecordProperty("mse_linear", metrics->mse);
    testing::Test::RecordProperty("rmse_linear", metrics->rmse);
    testing::Test::RecordProperty("psnr_display", psnr->psnr);
    testing::Test::RecordProperty(
        "preview_layout",
        "top=differential footprints;bottom=wavefront ray cones;left=mirror;right=eta-1.5");

    const auto status = write_png_preview(*preview, *output);
    ASSERT_TRUE(status.has_value()) << status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(*output));
#else
    FAIL() << "The ray-cone comparison preview requires the explicit PNG capability.";
#endif
}

} // namespace
} // namespace blackframe::renderer
