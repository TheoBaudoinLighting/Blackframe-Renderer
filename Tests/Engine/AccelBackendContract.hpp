#pragma once

#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine::test {

inline constexpr auto BackVisibility = renderer::RayMask{1U << 0U};
inline constexpr auto FrontVisibility = renderer::RayMask{1U << 1U};

[[nodiscard]] inline std::shared_ptr<const TriangleMesh> make_triangle_mesh(const float z) {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = 0.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 1.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 0.0F, .y = 1.0F, .z = z},
        },
        std::vector{
            renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
        },
        std::vector{
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        std::vector{TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] inline std::shared_ptr<const TriangleMesh>
make_triangle_mesh_with_leading_miss(const float z) {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = 2.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 3.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 2.0F, .y = 1.0F, .z = z},
            renderer::Point3{.x = 0.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 1.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 0.0F, .y = 1.0F, .z = z},
        },
        std::vector(6, renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}),
        std::vector{
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        std::vector{
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {3U, 4U, 5U}},
        });
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] inline std::shared_ptr<const TriangleMesh> make_coplanar_triangle_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = 0.0F, .y = 0.25F, .z = -2.0F},
            renderer::Point3{.x = 1.0F, .y = 0.25F, .z = -2.0F},
            renderer::Point3{.x = 0.0F, .y = 0.25F, .z = 2.0F},
        },
        std::vector(3, renderer::Normal3{.x = 0.0F, .y = -1.0F, .z = 0.0F}),
        std::vector{
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        std::vector{TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] inline std::array<AccelGeometry, 3> make_geometry_set() {
    return {
        AccelGeometry{
            .mesh = make_coplanar_triangle_mesh(),
            .instance = renderer::InstanceId{.value = 3U},
            .geometry = renderer::GeometryId{.value = 31U},
            .material = renderer::MaterialId{.value = 9U},
            .visibility_mask = renderer::AllRayVisibility,
        },
        AccelGeometry{
            .mesh = make_triangle_mesh(1.0F),
            .instance = renderer::InstanceId{.value = 5U},
            .geometry = renderer::GeometryId{.value = 41U},
            .material = renderer::MaterialId{.value = 11U},
            .visibility_mask = BackVisibility,
        },
        AccelGeometry{
            .mesh = make_triangle_mesh_with_leading_miss(0.0F),
            .instance = renderer::InstanceId{.value = 7U},
            .geometry = renderer::GeometryId{.value = 73U},
            .material = renderer::MaterialId{.value = 13U},
            .visibility_mask = FrontVisibility,
        },
    };
}

[[nodiscard]] inline renderer::Ray
make_ray(const renderer::Point3 origin, const renderer::Vector3 direction, const float t_min = 0.0F,
         const float t_max = std::numeric_limits<float>::infinity(),
         const renderer::RayMask mask = renderer::AllRayVisibility, const float time = 0.5F) {
    auto ray =
        renderer::Ray::create(origin, direction, t_min, t_max, time, mask, renderer::VacuumMedium);
    if (!ray) {
        throw std::runtime_error{ray.error().message};
    }
    return *ray;
}

inline void expect_backend_contract(const AccelBackend& backend,
                                    const AccelBackendKind expected_kind) {
    EXPECT_EQ(backend.kind(), expected_kind);

    const auto all_geometry_ray = make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                           renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    auto closest = backend.closest_hit(all_geometry_ray);
    ASSERT_TRUE(closest.has_value()) << closest.error().message;
    ASSERT_TRUE(closest->has_value());
    EXPECT_FLOAT_EQ(closest->value().triangle.parameter, 1.0F);
    EXPECT_EQ(closest->value().triangle.position,
              (renderer::Point3{.x = 0.25F, .y = 0.25F, .z = 0.0F}));
    EXPECT_EQ(closest->value().triangle.geometric_normal,
              (renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}));
    EXPECT_EQ(closest->value().triangle.barycentrics, (renderer::TriangleBarycentrics{
                                                          .vertex0 = 0.5F,
                                                          .vertex1 = 0.25F,
                                                          .vertex2 = 0.25F,
                                                      }));
    EXPECT_EQ(closest->value().identifiers, (renderer::SurfaceIdentifiers{
                                                .instance = renderer::InstanceId{.value = 7U},
                                                .geometry = renderer::GeometryId{.value = 73U},
                                                .primitive = renderer::PrimitiveId{.value = 1U},
                                                .material = renderer::MaterialId{.value = 13U},
                                            }));

    const auto back_only_ray = make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                        renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F,
                                        std::numeric_limits<float>::infinity(), BackVisibility);
    auto back_hit = backend.closest_hit(back_only_ray);
    ASSERT_TRUE(back_hit.has_value()) << back_hit.error().message;
    ASSERT_TRUE(back_hit->has_value());
    EXPECT_FLOAT_EQ(back_hit->value().triangle.parameter, 2.0F);
    EXPECT_EQ(back_hit->value().identifiers.geometry, renderer::GeometryId{.value = 41U});
    EXPECT_EQ(back_hit->value().identifiers.instance, renderer::InstanceId{.value = 5U});

    const auto lateral_miss = make_ray(renderer::Point3{.x = 1.25F, .y = 1.25F, .z = -1.0F},
                                       renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    auto miss = backend.closest_hit(lateral_miss);
    ASSERT_TRUE(miss.has_value()) << miss.error().message;
    EXPECT_FALSE(miss->has_value());

    const auto clipped = make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                  renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F, 0.75F);
    auto clipped_hit = backend.closest_hit(clipped);
    ASSERT_TRUE(clipped_hit.has_value()) << clipped_hit.error().message;
    EXPECT_FALSE(clipped_hit->has_value());

    auto visible_occlusion = backend.occluded(all_geometry_ray);
    ASSERT_TRUE(visible_occlusion.has_value()) << visible_occlusion.error().message;
    EXPECT_TRUE(*visible_occlusion);

    auto clipped_occlusion = backend.occluded(clipped);
    ASSERT_TRUE(clipped_occlusion.has_value()) << clipped_occlusion.error().message;
    EXPECT_FALSE(*clipped_occlusion);

    const auto masked =
        make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                 renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F,
                 std::numeric_limits<float>::infinity(), renderer::RayMask{1U << 8U});
    auto masked_hit = backend.closest_hit(masked);
    ASSERT_TRUE(masked_hit.has_value()) << masked_hit.error().message;
    EXPECT_FALSE(masked_hit->has_value());
    auto masked_occlusion = backend.occluded(masked);
    ASSERT_TRUE(masked_occlusion.has_value()) << masked_occlusion.error().message;
    EXPECT_FALSE(*masked_occlusion);

    const auto invalid_time =
        make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                 renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 0.0F,
                 std::numeric_limits<float>::infinity(), renderer::AllRayVisibility, 1.25F);
    const auto invalid_closest = backend.closest_hit(invalid_time);
    ASSERT_FALSE(invalid_closest.has_value());
    EXPECT_EQ(invalid_closest.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(invalid_closest.error().message.empty());
    const auto invalid_occlusion = backend.occluded(invalid_time);
    ASSERT_FALSE(invalid_occlusion.has_value());
    EXPECT_EQ(invalid_occlusion.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(invalid_occlusion.error().message.empty());
}

inline void expect_empty_backend(const AccelBackend& backend,
                                 const AccelBackendKind expected_kind) {
    EXPECT_EQ(backend.kind(), expected_kind);
    const auto ray = make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                              renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    const auto hit = backend.closest_hit(ray);
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    EXPECT_FALSE(hit->has_value());
    const auto shadowed = backend.occluded(ray);
    ASSERT_TRUE(shadowed.has_value()) << shadowed.error().message;
    EXPECT_FALSE(*shadowed);
}

} // namespace blackframe::engine::test
