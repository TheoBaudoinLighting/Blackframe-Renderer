#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
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

template <GeometryScalar Scalar>
[[nodiscard]] RayT<Scalar> test_ray_t(const Point3T<Scalar> origin,
                                      const Vector3T<Scalar> direction,
                                      const Scalar time = Scalar{0.25}) {
    return RayT<Scalar>::create(origin, direction, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), time, 0x55AA55AAU,
                                MediumId{.value = 3U})
        .value();
}

[[nodiscard]] Ray test_ray(const Point3 origin, const Vector3 direction) {
    return test_ray_t<float>(origin, direction);
}

[[nodiscard]] SurfaceInteraction planar_surface(const Vector3 dpdu = Vector3{.x = 1.0F},
                                                const Vector3 dpdv = Vector3{.y = 1.0F}) {
    return SurfaceInteraction::create(Point3{.z = -2.0F}, Normal3{.z = 1.0F}, Normal3{.z = 1.0F},
                                      Point2{}, dpdu, dpdv, SurfaceIdentifiers{}, 0.25F)
        .value();
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

[[nodiscard]] core::Result<Point3> intersect_z_plane(const Point3 origin, const Vector3 direction,
                                                     const float plane_z) {
    if (!std::isfinite(direction.z) || direction.z == 0.0F) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The preview ray is parallel to its target plane.",
        });
    }
    const auto parameter = (plane_z - origin.z) / direction.z;
    const auto point = origin + direction * parameter;
    if (!std::isfinite(parameter) || !std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The preview target-plane intersection is not representable.",
        });
    }
    return point;
}

[[nodiscard]] float filtered_checker(const Point3 center,
                                     const SurfacePointDifferentials footprint) {
    constexpr auto frequency = 42.0F;
    constexpr auto samples_per_axis = 4U;
    auto sum = 0.0F;
    for (auto sample_y = std::uint32_t{}; sample_y < samples_per_axis; ++sample_y) {
        for (auto sample_x = std::uint32_t{}; sample_x < samples_per_axis; ++sample_x) {
            const auto offset_x =
                (static_cast<float>(sample_x) + 0.5F) / static_cast<float>(samples_per_axis) - 0.5F;
            const auto offset_y =
                (static_cast<float>(sample_y) + 0.5F) / static_cast<float>(samples_per_axis) - 0.5F;
            const auto sample = center + footprint.dpdx * offset_x + footprint.dpdy * offset_y;
            const auto cell_x = static_cast<std::int64_t>(std::floor(sample.x * frequency));
            const auto cell_y = static_cast<std::int64_t>(std::floor(sample.y * frequency));
            sum += ((cell_x ^ cell_y) & 1LL) == 0LL ? 0.92F : 0.06F;
        }
    }
    return sum / static_cast<float>(samples_per_axis * samples_per_axis);
}

#if BLACKFRAME_HAS_PNG_PREVIEW
[[nodiscard]] core::Status
write_specular_differential_preview(const std::filesystem::path& output) {
    constexpr auto extent = RenderExtent{.width = 800U, .height = 800U};
    const auto frame =
        OrthonormalFrame::from_normal_and_tangent(Normal3{.z = 1.0F}, Vector3{.x = 1.0F});
    if (!frame) {
        return std::unexpected(frame.error());
    }
    const auto camera = PinholeCamera::create(
        Point3{.z = 2.0F}, *frame, extent, std::numbers::pi_v<float> / 3.0F, 0.0F,
        std::numeric_limits<float>::infinity(), AllRayVisibility, VacuumMedium);
    if (!camera) {
        return std::unexpected(camera.error());
    }
    auto film = Film::create(extent);
    if (!film) {
        return std::unexpected(film.error());
    }
    constexpr auto normal = Normal3{.z = 1.0F};
    for (auto y = std::uint32_t{}; y < extent.height; ++y) {
        for (auto x = std::uint32_t{}; x < extent.width; ++x) {
            const auto camera_differential = camera->generate_primary_ray_differential(
                Point2{.x = static_cast<float>(x) + 0.5F, .y = static_cast<float>(y) + 0.5F}, 0.0F);
            if (!camera_differential) {
                return std::unexpected(camera_differential.error());
            }
            const auto interface_position = intersect_z_plane(
                camera_differential->ray().origin(), camera_differential->ray().direction(), 0.0F);
            if (!interface_position) {
                return std::unexpected(interface_position.error());
            }
            const auto interface = SurfaceInteraction::create(
                *interface_position, normal, normal, Point2{}, Vector3{.x = 1.0F},
                Vector3{.y = 1.0F}, SurfaceIdentifiers{}, 0.0F);
            if (!interface) {
                return std::unexpected(interface.error());
            }
            const auto interface_footprint =
                surface_point_differentials(*camera_differential, *interface);
            if (!interface_footprint) {
                return std::unexpected(interface_footprint.error());
            }
            const auto rx_position = *interface_position + interface_footprint->dpdx;
            const auto ry_position = *interface_position + interface_footprint->dpdy;

            auto next_direction = Vector3{};
            auto next_differential = std::optional<RayDifferential>{};
            auto target_z = 0.0F;
            const auto mirror_panel = x < extent.width / 2U;
            if (mirror_panel) {
                const auto incident = camera_differential->ray().direction();
                next_direction = incident - 2.0F * dot(incident, normal) * Vector3{.z = 1.0F};
                target_z = 4.0F;
                const auto next_ray = Ray::create(*interface_position, next_direction, 0.0F,
                                                  std::numeric_limits<float>::infinity(), 0.0F,
                                                  AllRayVisibility, VacuumMedium);
                if (!next_ray) {
                    return std::unexpected(next_ray.error());
                }
                const auto reflected =
                    propagate_specular_reflection(*camera_differential, *next_ray, normal,
                                                  rx_position, normal, ry_position, normal);
                if (!reflected) {
                    return std::unexpected(reflected.error());
                }
                next_differential = *reflected;
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
                const auto next_ray = Ray::create(*interface_position, next_direction, 0.0F,
                                                  std::numeric_limits<float>::infinity(), 0.0F,
                                                  AllRayVisibility, VacuumMedium);
                if (!next_ray) {
                    return std::unexpected(next_ray.error());
                }
                const auto transmitted = propagate_specular_transmission(
                    *camera_differential, *next_ray, normal, rx_position, normal, ry_position,
                    normal, true, 1.0F, 1.5F);
                if (!transmitted) {
                    return std::unexpected(transmitted.error());
                }
                if (!*transmitted) {
                    return std::unexpected(core::Error{
                        .code = core::StatusCode::invalid_argument,
                        .message = "The entering preview differential unexpectedly reached TIR.",
                    });
                }
                next_differential = **transmitted;
            }

            const auto target_position =
                intersect_z_plane(interface_position.value(), next_direction, target_z);
            if (!target_position) {
                return std::unexpected(target_position.error());
            }
            const auto target = SurfaceInteraction::create(
                *target_position, normal, normal, Point2{}, Vector3{.x = 1.0F}, Vector3{.y = 1.0F},
                SurfaceIdentifiers{}, 0.0F);
            if (!target) {
                return std::unexpected(target.error());
            }
            const auto target_footprint = surface_point_differentials(*next_differential, *target);
            if (!target_footprint) {
                return std::unexpected(target_footprint.error());
            }
            const auto checker = filtered_checker(*target_position, *target_footprint);
            const auto normalized_x = static_cast<float>(x) / static_cast<float>(extent.width - 1U);
            const auto normalized_y =
                static_cast<float>(y) / static_cast<float>(extent.height - 1U);
            const auto vignette =
                1.0F - 0.24F * std::hypot(2.0F * normalized_x - 1.0F, 2.0F * normalized_y - 1.0F);
            auto color = mirror_panel ? LinearRGB{.red = checker * 0.94F,
                                                  .green = checker * 0.76F,
                                                  .blue = checker * 0.45F}
                                      : LinearRGB{.red = checker * 0.42F,
                                                  .green = checker * 0.78F,
                                                  .blue = checker * 0.96F};
            color.red *= vignette;
            color.green *= vignette;
            color.blue *= vignette;
            if (x == extent.width / 2U || x + 1U == extent.width / 2U) {
                color = LinearRGB{.red = 0.95F, .green = 0.95F, .blue = 0.95F};
            }
            const auto status = film->add_sample(x, y, color, 1.0F);
            if (!status) {
                return status;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(RayDifferentialTest, KeepsTheTraversalRayAbiSeparateAndRejectsInvalidAuxiliaries) {
    static_assert(!std::is_same_v<RayDifferential, Ray>);
    static_assert(std::is_standard_layout_v<RayDifferential>);
    static_assert(std::is_trivially_copyable_v<RayDifferential>);
    static_assert(std::is_standard_layout_v<ReferenceRayDifferential>);
    static_assert(std::is_trivially_copyable_v<ReferenceRayDifferential>);
    EXPECT_EQ(sizeof(RayDifferential), sizeof(Ray) + 2U * sizeof(Point3) + 2U * sizeof(Vector3));

    const auto ray = test_ray(Point3{}, Vector3{.z = -1.0F});
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(RayDifferential::create(ray, Point3{}, Vector3{}, Point3{}, Vector3{.z = -1.0F})
                     .has_value());
    EXPECT_FALSE(RayDifferential::create(ray, Point3{.x = nan}, Vector3{.z = -1.0F}, Point3{},
                                         Vector3{.z = -1.0F})
                     .has_value());
}

TEST(RayDifferentialTest, RecoversSignedPlanarPositionAndUvDerivatives) {
    const auto ray = test_ray(Point3{}, Vector3{.z = -1.0F});
    const auto differential = RayDifferential::create(ray, Point3{}, Vector3{.x = 0.5F, .z = -1.0F},
                                                      Point3{}, Vector3{.y = -0.5F, .z = -1.0F});
    ASSERT_TRUE(differential.has_value());

    const auto positions = surface_point_differentials(*differential, planar_surface());
    ASSERT_TRUE(positions.has_value()) << positions.error().message;
    EXPECT_EQ(positions->dpdx, (Vector3{.x = 1.0F}));
    EXPECT_EQ(positions->dpdy, (Vector3{.y = -1.0F}));
    const auto texture = texture_coordinate_differentials(planar_surface(), *positions);
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    EXPECT_EQ(*texture, (TextureCoordinateDifferentials{
                            .dudx = 1.0F,
                            .dvdx = 0.0F,
                            .dudy = 0.0F,
                            .dvdy = -1.0F,
                        }));
}

TEST(RayDifferentialTest, UsesGeometricPlaneAndRejectsParallelOrSingularRequests) {
    const auto tilted_shading_normal = Normal3{.x = 0.6F, .z = 0.8F};
    const auto surface = SurfaceInteraction::create(
        Point3{.z = -2.0F}, Normal3{.z = 1.0F}, tilted_shading_normal, Point2{}, Vector3{.x = 1.0F},
        Vector3{.y = 1.0F}, SurfaceIdentifiers{}, 0.25F);
    ASSERT_TRUE(surface.has_value());
    const auto ray = test_ray(Point3{}, Vector3{.z = -1.0F});
    const auto differential = RayDifferential::create(ray, Point3{}, Vector3{.x = 0.5F, .z = -1.0F},
                                                      Point3{}, Vector3{.y = -0.5F, .z = -1.0F});
    ASSERT_TRUE(differential.has_value());
    const auto positions = surface_point_differentials(*differential, *surface);
    ASSERT_TRUE(positions.has_value());
    EXPECT_EQ(positions->dpdx, (Vector3{.x = 1.0F}));

    const auto parallel = RayDifferential::create(ray, Point3{}, Vector3{.x = 1.0F}, Point3{},
                                                  Vector3{.y = 1.0F, .z = -1.0F});
    ASSERT_TRUE(parallel.has_value());
    const auto parallel_result = surface_point_differentials(*parallel, *surface);
    ASSERT_FALSE(parallel_result.has_value());
    EXPECT_EQ(parallel_result.error().code, core::StatusCode::invalid_argument);

    const auto singular_surface = planar_surface(Vector3{.x = 1.0F}, Vector3{.x = 2.0F});
    const auto singular = texture_coordinate_differentials(singular_surface, *positions);
    ASSERT_FALSE(singular.has_value());
    EXPECT_EQ(singular.error().code, core::StatusCode::invalid_argument);
}

TEST(RayDifferentialTest, PropagatesIdealReflectionAndTransmissionWithoutFallback) {
    constexpr auto incident_x = 0.2F;
    const auto incident_z = -std::sqrt(1.0F - incident_x * incident_x);
    const auto primary = test_ray(Point3{.z = 1.0F}, Vector3{.z = -1.0F});
    const auto differential = RayDifferential::create(
        primary, Point3{.z = 1.0F}, Vector3{.x = incident_x, .z = incident_z}, Point3{.z = 1.0F},
        Vector3{.y = incident_x, .z = incident_z});
    ASSERT_TRUE(differential.has_value());
    const auto reflected_ray = test_ray(Point3{.z = 0.001F}, Vector3{.z = 1.0F});
    const auto reflected = propagate_specular_reflection(
        *differential, reflected_ray, Normal3{.z = 1.0F}, Point3{.x = 0.2F}, Normal3{.z = 1.0F},
        Point3{.y = 0.2F}, Normal3{.z = 1.0F});
    ASSERT_TRUE(reflected.has_value()) << reflected.error().message;
    EXPECT_NEAR(reflected->rx_direction().x, incident_x, 2.0e-6F);
    EXPECT_NEAR(reflected->rx_direction().z, -incident_z, 2.0e-6F);
    EXPECT_NEAR(reflected->ry_direction().y, incident_x, 2.0e-6F);
    EXPECT_EQ(reflected->rx_origin(), (Point3{.x = 0.2F}));
    EXPECT_EQ(reflected->ray().direction(), reflected_ray.direction());

    const auto transmitted_ray = test_ray(Point3{.z = -0.001F}, Vector3{.z = -1.0F});
    const auto transmitted = propagate_specular_transmission(
        *differential, transmitted_ray, Normal3{.z = 1.0F}, Point3{.x = 0.2F}, Normal3{.z = 1.0F},
        Point3{.y = 0.2F}, Normal3{.z = 1.0F}, true, 1.0F, 1.5F);
    ASSERT_TRUE(transmitted.has_value()) << transmitted.error().message;
    ASSERT_TRUE(transmitted->has_value());
    EXPECT_NEAR((**transmitted).rx_direction().x, incident_x / 1.5F, 2.0e-6F);
    EXPECT_NEAR((**transmitted).rx_direction().z,
                -std::sqrt(1.0F - (incident_x / 1.5F) * (incident_x / 1.5F)), 2.0e-6F);

    const auto equal_eta = propagate_specular_transmission(
        *differential, transmitted_ray, Normal3{.z = 1.0F}, Point3{.x = 0.2F}, Normal3{.z = 1.0F},
        Point3{.y = 0.2F}, Normal3{.z = 1.0F}, true, 1.0F, 1.0F);
    ASSERT_TRUE(equal_eta.has_value());
    ASSERT_TRUE(equal_eta->has_value());
    EXPECT_NEAR((**equal_eta).rx_direction().x, differential->rx_direction().x, 2.0e-6F);
    EXPECT_NEAR((**equal_eta).rx_direction().z, differential->rx_direction().z, 2.0e-6F);
    EXPECT_NEAR((**equal_eta).ry_direction().y, differential->ry_direction().y, 2.0e-6F);
    EXPECT_NEAR((**equal_eta).ry_direction().z, differential->ry_direction().z, 2.0e-6F);
}

template <GeometryScalar Scalar> void expect_equal_eta_near_tangent_transmission() {
    const auto grazing_cosine = std::sqrt(Scalar{64} * std::numeric_limits<Scalar>::epsilon());
    const auto incident_direction = Vector3T<Scalar>{.x = Scalar{1}, .z = -grazing_cosine};
    const auto incident_ray =
        test_ray_t<Scalar>(Point3T<Scalar>{.z = Scalar{1}}, incident_direction);
    const auto differential =
        RayDifferentialT<Scalar>::create(incident_ray, incident_ray.origin(), incident_direction,
                                         incident_ray.origin(), incident_direction);
    ASSERT_TRUE(differential.has_value());
    const auto transmitted_ray =
        test_ray_t<Scalar>(Point3T<Scalar>{.z = Scalar{-0.001}}, incident_direction);

    const auto propagated = propagate_specular_transmission(
        *differential, transmitted_ray, Normal3T<Scalar>{.z = Scalar{1}}, Point3T<Scalar>{},
        Normal3T<Scalar>{.z = Scalar{1}}, Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, true,
        Scalar{1}, Scalar{1});

    ASSERT_TRUE(propagated.has_value()) << propagated.error().message;
    ASSERT_TRUE(propagated->has_value());
    EXPECT_EQ((**propagated).rx_direction(), incident_direction);
    EXPECT_EQ((**propagated).ry_direction(), incident_direction);
}

TEST(RayDifferentialTest, MatchesEqualEtaNearTangentInBothPrecisions) {
    expect_equal_eta_near_tangent_transmission<float>();
    expect_equal_eta_near_tangent_transmission<double>();
}

TEST(RayDifferentialTest, PropagatesSubcriticalExitingTransmissionWithSnell) {
    constexpr auto central_tangent = 0.2F;
    constexpr auto auxiliary_tangent = 0.25F;
    const auto incident_ray = test_ray(
        Point3{.z = -1.0F},
        Vector3{.x = central_tangent, .z = std::sqrt(1.0F - central_tangent * central_tangent)});
    const auto differential = RayDifferential::create(
        incident_ray, incident_ray.origin(),
        Vector3{.x = auxiliary_tangent,
                .z = std::sqrt(1.0F - auxiliary_tangent * auxiliary_tangent)},
        incident_ray.origin(),
        Vector3{.y = auxiliary_tangent,
                .z = std::sqrt(1.0F - auxiliary_tangent * auxiliary_tangent)});
    ASSERT_TRUE(differential.has_value());
    constexpr auto central_transmitted_tangent = 1.5F * central_tangent;
    const auto transmitted_ray = test_ray(
        Point3{.z = 0.001F},
        Vector3{.x = central_transmitted_tangent,
                .z = std::sqrt(1.0F - central_transmitted_tangent * central_transmitted_tangent)});

    const auto propagated = propagate_specular_transmission(
        *differential, transmitted_ray, Normal3{.z = 1.0F}, Point3{}, Normal3{.z = 1.0F}, Point3{},
        Normal3{.z = 1.0F}, false, 1.0F, 1.5F);

    ASSERT_TRUE(propagated.has_value()) << propagated.error().message;
    ASSERT_TRUE(propagated->has_value());
    constexpr auto auxiliary_transmitted_tangent = 1.5F * auxiliary_tangent;
    EXPECT_NEAR((**propagated).rx_direction().x, auxiliary_transmitted_tangent, 2.0e-6F);
    EXPECT_NEAR((**propagated).rx_direction().z,
                std::sqrt(1.0F - auxiliary_transmitted_tangent * auxiliary_transmitted_tangent),
                2.0e-6F);
}

TEST(RayDifferentialTest, RejectsIncoherentCentralSpecularRays) {
    const auto incident_ray = test_ray(Point3{.z = 1.0F}, Vector3{.z = -1.0F});
    const auto differential = RayDifferential::create(
        incident_ray, incident_ray.origin(), Vector3{.x = 0.1F, .z = -0.99498743F},
        incident_ray.origin(), Vector3{.y = 0.1F, .z = -0.99498743F});
    ASSERT_TRUE(differential.has_value());
    const auto wrong_direction = test_ray(Point3{}, Vector3{.x = 1.0F});
    const auto wrong_time = test_ray_t<float>(Point3{}, Vector3{.z = 1.0F}, 0.5F);

    const auto direction_error =
        propagate_specular_reflection(*differential, wrong_direction, Normal3{.z = 1.0F}, Point3{},
                                      Normal3{.z = 1.0F}, Point3{}, Normal3{.z = 1.0F});
    const auto time_error =
        propagate_specular_reflection(*differential, wrong_time, Normal3{.z = 1.0F}, Point3{},
                                      Normal3{.z = 1.0F}, Point3{}, Normal3{.z = 1.0F});

    ASSERT_FALSE(direction_error.has_value());
    EXPECT_EQ(direction_error.error().code, core::StatusCode::invalid_argument);
    ASSERT_FALSE(time_error.has_value());
    EXPECT_EQ(time_error.error().code, core::StatusCode::invalid_argument);
}

TEST(RayDifferentialTest, ReportsAuxiliaryTotalInternalReflectionAsADiscontinuity) {
    const auto primary = test_ray(Point3{.z = -1.0F}, Vector3{.z = 1.0F});
    const auto differential =
        RayDifferential::create(primary, Point3{.z = -1.0F}, Vector3{.x = 0.8F, .z = 0.6F},
                                Point3{.z = -1.0F}, Vector3{.y = 0.8F, .z = 0.6F});
    ASSERT_TRUE(differential.has_value());
    const auto transmitted_ray = test_ray(Point3{.z = 0.001F}, Vector3{.z = 1.0F});
    const auto propagated = propagate_specular_transmission(
        *differential, transmitted_ray, Normal3{.z = 1.0F}, Point3{}, Normal3{.z = 1.0F}, Point3{},
        Normal3{.z = 1.0F}, false, 1.0F, 1.5F);
    ASSERT_TRUE(propagated.has_value()) << propagated.error().message;
    EXPECT_FALSE(propagated->has_value());
}

TEST(RayDifferentialTest, WritesStableMirrorAndTransmissionFootprintPreview) {
    const auto output = checksum_output_path();
    if (!output) {
        GTEST_SKIP() << "The explicit PNG checksum output path was not supplied.";
    }
#if BLACKFRAME_HAS_PNG_PREVIEW
    const auto status = write_specular_differential_preview(*output);
    ASSERT_TRUE(status.has_value()) << status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(*output));
    testing::Test::RecordProperty(
        "preview_layout",
        "left=mirror differential footprint;right=eta-1.5 transmission differential footprint");
#else
    FAIL() << "The ray-differential preview requires the explicit PNG capability.";
#endif
}

} // namespace
} // namespace blackframe::renderer
