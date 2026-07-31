#pragma once

#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine::test {

inline constexpr auto BackVisibility = renderer::RayMask{1U << 0U};
inline constexpr auto FrontVisibility = renderer::RayMask{1U << 1U};
inline constexpr auto NearShadowVisibility = renderer::RayMask{1U << 2U};
inline constexpr auto FarShadowVisibility = renderer::RayMask{1U << 3U};
inline constexpr auto HierarchyVisibility = renderer::RayMask{1U << 4U};

[[nodiscard]] inline renderer::Matrix4 identity_transform_matrix() {
    return renderer::identity_matrix<renderer::TransportScalar>();
}

[[nodiscard]] inline renderer::Matrix4 translation_matrix(const renderer::Vector3 offset) {
    auto matrix = identity_transform_matrix();
    matrix(0, 3) = offset.x;
    matrix(1, 3) = offset.y;
    matrix(2, 3) = offset.z;
    return matrix;
}

[[nodiscard]] inline renderer::Matrix4 scale_matrix(const renderer::Vector3 factors) {
    auto matrix = identity_transform_matrix();
    matrix(0, 0) = factors.x;
    matrix(1, 1) = factors.y;
    matrix(2, 2) = factors.z;
    return matrix;
}

[[nodiscard]] inline renderer::Matrix4 positive_quarter_turn_z_matrix() {
    auto matrix = identity_transform_matrix();
    matrix(0, 0) = 0.0F;
    matrix(0, 1) = -1.0F;
    matrix(1, 0) = 1.0F;
    matrix(1, 1) = 0.0F;
    return matrix;
}

[[nodiscard]] inline std::shared_ptr<const TriangleMesh> make_triangle_mesh(const float z) {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = 0.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 1.0F, .y = 0.0F, .z = z},
            renderer::Point3{.x = 0.0F, .y = 1.0F, .z = z},
        },
        std::vector(3, renderer::Normal3{.z = 1.0F}),
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
        std::vector(6, renderer::Normal3{.z = 1.0F}),
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
        std::vector(3, renderer::Normal3{.y = -1.0F}),
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

[[nodiscard]] inline FrameSceneHandle make_basic_scene() {
    auto scene =
        FrameScene::create(
            FrameSceneDescription{
                .objects =
                    {
                        SceneObject{.id = {.value = 103U}},
                        SceneObject{.id = {.value = 105U}},
                        SceneObject{.id = {.value = 107U}},
                    },
                .geometries =
                    {
                        SceneGeometry{.id = {.value = 31U}, .mesh = make_coplanar_triangle_mesh()},
                        SceneGeometry{.id = {.value = 41U}, .mesh = make_triangle_mesh(1.0F)},
                        SceneGeometry{
                            .id = {.value = 73U},
                            .mesh = make_triangle_mesh_with_leading_miss(0.0F),
                        },
                    },
                .materials =
                    {
                        SceneMaterial{.id = {.value = 9U}},
                        SceneMaterial{.id = {.value = 11U}},
                        SceneMaterial{.id = {.value = 13U}},
                    },
                .instances =
                    {
                        SceneInstance{
                            .id = {.value = 3U},
                            .parent = std::nullopt,
                            .object = {.value = 103U},
                            .geometry = {.value = 31U},
                            .material = {.value = 9U},
                            .local_to_parent = identity_transform_matrix(),
                        },
                        SceneInstance{
                            .id = {.value = 5U},
                            .parent = std::nullopt,
                            .object = {.value = 105U},
                            .geometry = {.value = 41U},
                            .material = {.value = 11U},
                            .local_to_parent = identity_transform_matrix(),
                            .visibility_mask = BackVisibility,
                        },
                        SceneInstance{
                            .id = {.value = 7U},
                            .parent = std::nullopt,
                            .object = {.value = 107U},
                            .geometry = {.value = 73U},
                            .material = {.value = 13U},
                            .local_to_parent = identity_transform_matrix(),
                            .visibility_mask = FrontVisibility,
                        },
                    },
            });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] inline FrameSceneHandle make_empty_scene() {
    auto scene = FrameScene::create(FrameSceneDescription{});
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] inline std::shared_ptr<const TriangleMesh> make_instanced_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{.x = 4.0F},
            renderer::Point3{.x = 5.0F},
            renderer::Point3{.x = 4.0F, .y = 1.0F},
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.y = 1.0F, .z = 1.0F},
        },
        std::vector(6, renderer::Normal3{.x = 1.0F}),
        std::vector{
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
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

[[nodiscard]] inline FrameSceneHandle make_instanced_scene() {
    const auto geometry_id = renderer::GeometryId{.value = 84U};
    const auto scale = scale_matrix(renderer::Vector3{.x = -2.0F, .y = 3.0F, .z = 4.0F});
    const auto rotation = positive_quarter_turn_z_matrix();
    const auto far_transform =
        translation_matrix(renderer::Vector3{.x = 10.0F, .z = 8.0F}) * rotation * scale;
    auto
        scene =
            FrameScene::create(
                FrameSceneDescription{
                    .objects =
                        {
                            SceneObject{.id = {.value = 110U}},
                            SceneObject{.id = {.value = 120U}},
                            SceneObject{.id = {.value = 130U}},
                            SceneObject{.id = {.value = 140U}},
                        },
                    .geometries = {SceneGeometry{.id = geometry_id, .mesh = make_instanced_mesh()}},
                    .materials =
                        {
                            SceneMaterial{.id = {.value = 210U}},
                            SceneMaterial{.id = {.value = 220U}},
                            SceneMaterial{.id = {.value = 230U}},
                            SceneMaterial{.id = {.value = 240U}},
                        },
                    // Deliberately neither insertion- nor identifier-topological.
                    .instances =
                        {
                            SceneInstance{
                                .id = {.value = 20U},
                                .parent = renderer::InstanceId{.value = 10U},
                                .object = {.value = 120U},
                                .geometry = geometry_id,
                                .material = {.value = 220U},
                                .local_to_parent = scale,
                                .visibility_mask = NearShadowVisibility,
                            },
                            SceneInstance{
                                .id = {.value = 40U},
                                .parent = std::nullopt,
                                .object = {.value = 140U},
                                .geometry = geometry_id,
                                .material = {.value = 240U},
                                .local_to_parent = far_transform,
                                .visibility_mask = FarShadowVisibility,
                            },
                            SceneInstance{
                                .id = {.value = 30U},
                                .parent = std::nullopt,
                                .object = {.value = 130U},
                                .geometry = geometry_id,
                                .material = {.value = 230U},
                                .local_to_parent =
                                    translation_matrix(renderer::Vector3{.x = 10.0F}),
                                .visibility_mask = HierarchyVisibility,
                            },
                            SceneInstance{
                                .id = {.value = 10U},
                                .parent = renderer::InstanceId{.value = 30U},
                                .object = {.value = 110U},
                                .geometry = geometry_id,
                                .material = {.value = 210U},
                                .local_to_parent = rotation,
                                .visibility_mask = HierarchyVisibility,
                            },
                        },
                });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] inline FrameSceneHandle make_unrepresentable_instance_scene() {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{.id = {.value = 2U}, .mesh = make_triangle_mesh(0.0F)}},
        .materials = {SceneMaterial{.id = {.value = 3U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 4U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 2U},
                    .material = {.value = 3U},
                    .local_to_parent = translation_matrix(renderer::Vector3{.x = 16'777'216.0F}),
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
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
                                           renderer::Vector3{.z = 1.0F});
    auto closest = backend.closest_hit(all_geometry_ray);
    ASSERT_TRUE(closest.has_value()) << closest.error().message;
    ASSERT_TRUE(closest->has_value());
    EXPECT_EQ(closest->value().object, (renderer::ObjectId{.value = 107U}));
    EXPECT_FLOAT_EQ(closest->value().triangle.parameter, 1.0F);
    EXPECT_EQ(closest->value().triangle.position,
              (renderer::Point3{.x = 0.25F, .y = 0.25F, .z = 0.0F}));
    EXPECT_EQ(closest->value().triangle.geometric_normal, (renderer::Normal3{.z = 1.0F}));
    EXPECT_EQ(closest->value().triangle.barycentrics, (renderer::TriangleBarycentrics{
                                                          .vertex0 = 0.5F,
                                                          .vertex1 = 0.25F,
                                                          .vertex2 = 0.25F,
                                                      }));
    EXPECT_EQ(closest->value().identifiers, (renderer::SurfaceIdentifiers{
                                                .instance = {.value = 7U},
                                                .geometry = {.value = 73U},
                                                .primitive = {.value = 1U},
                                                .material = {.value = 13U},
                                            }));

    const auto back_only_ray =
        make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F}, renderer::Vector3{.z = 1.0F},
                 0.0F, std::numeric_limits<float>::infinity(), BackVisibility);
    auto back_hit = backend.closest_hit(back_only_ray);
    ASSERT_TRUE(back_hit.has_value()) << back_hit.error().message;
    ASSERT_TRUE(back_hit->has_value());
    EXPECT_FLOAT_EQ(back_hit->value().triangle.parameter, 2.0F);
    EXPECT_EQ(back_hit->value().identifiers.geometry, (renderer::GeometryId{.value = 41U}));
    EXPECT_EQ(back_hit->value().identifiers.instance, (renderer::InstanceId{.value = 5U}));

    const auto lateral_miss = make_ray(renderer::Point3{.x = 1.25F, .y = 1.25F, .z = -1.0F},
                                       renderer::Vector3{.z = 1.0F});
    auto miss = backend.closest_hit(lateral_miss);
    ASSERT_TRUE(miss.has_value()) << miss.error().message;
    EXPECT_FALSE(miss->has_value());

    const auto clipped = make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                  renderer::Vector3{.z = 1.0F}, 0.0F, 0.75F);
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
        make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F}, renderer::Vector3{.z = 1.0F},
                 0.0F, std::numeric_limits<float>::infinity(), renderer::RayMask{1U << 8U});
    auto masked_hit = backend.closest_hit(masked);
    ASSERT_TRUE(masked_hit.has_value()) << masked_hit.error().message;
    EXPECT_FALSE(masked_hit->has_value());
    auto masked_occlusion = backend.occluded(masked);
    ASSERT_TRUE(masked_occlusion.has_value()) << masked_occlusion.error().message;
    EXPECT_FALSE(*masked_occlusion);

    const auto invalid_time =
        make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F}, renderer::Vector3{.z = 1.0F},
                 0.0F, std::numeric_limits<float>::infinity(), renderer::AllRayVisibility, 1.25F);
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
                              renderer::Vector3{.z = 1.0F});
    const auto hit = backend.closest_hit(ray);
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    EXPECT_FALSE(hit->has_value());
    const auto shadowed = backend.occluded(ray);
    ASSERT_TRUE(shadowed.has_value()) << shadowed.error().message;
    EXPECT_FALSE(*shadowed);
}

[[nodiscard]] inline std::array<renderer::Ray, 2> instanced_scene_rays() {
    return {
        make_ray(renderer::Point3{.x = 9.25F, .y = -0.5F, .z = -7.0F},
                 renderer::Vector3{.z = 2.0F}),
        make_ray(renderer::Point3{.x = 9.25F, .y = -0.5F, .z = -7.0F}, renderer::Vector3{.z = 2.0F},
                 5.0F),
    };
}

struct OcclusionExpectation final {
    const char* name;
    renderer::Ray ray;
    bool expected;
};

[[nodiscard]] inline std::array<OcclusionExpectation, 10> instanced_scene_occlusion_expectations() {
    const auto origin = renderer::Point3{.x = 9.25F, .y = -0.5F, .z = -7.0F};
    const auto direction = renderer::Vector3{.z = 2.0F};
    const auto infinity = std::numeric_limits<float>::infinity();
    constexpr auto unrelated_visibility = renderer::RayMask{1U << 8U};

    return {
        OcclusionExpectation{
            .name = "all visibility reaches the near blocker",
            .ray = make_ray(origin, direction),
            .expected = true,
        },
        OcclusionExpectation{
            .name = "near visibility selects the near blocker",
            .ray = make_ray(origin, direction, 0.0F, infinity, NearShadowVisibility),
            .expected = true,
        },
        OcclusionExpectation{
            .name = "far visibility skips near and reaches far",
            .ray = make_ray(origin, direction, 0.0F, infinity, FarShadowVisibility),
            .expected = true,
        },
        OcclusionExpectation{
            .name = "combined visibility reaches an eligible blocker",
            .ray = make_ray(origin, direction, 0.0F, infinity,
                            NearShadowVisibility | FarShadowVisibility),
            .expected = true,
        },
        OcclusionExpectation{
            .name = "unrelated visibility sees no blocker",
            .ray = make_ray(origin, direction, 0.0F, infinity, unrelated_visibility),
            .expected = false,
        },
        OcclusionExpectation{
            .name = "zero visibility sees no instance",
            .ray = make_ray(origin, direction, 0.0F, infinity, renderer::RayMask{}),
            .expected = false,
        },
        OcclusionExpectation{
            .name = "finite segment includes the near blocker",
            .ray = make_ray(origin, direction, 0.0F, 6.0F, NearShadowVisibility),
            .expected = true,
        },
        OcclusionExpectation{
            .name = "finite segment excludes the far blocker",
            .ray = make_ray(origin, direction, 0.0F, 6.0F, FarShadowVisibility),
            .expected = false,
        },
        OcclusionExpectation{
            .name = "late segment excludes the near blocker",
            .ray = make_ray(origin, direction, 5.0F, infinity, NearShadowVisibility),
            .expected = false,
        },
        OcclusionExpectation{
            .name = "late segment reaches the far blocker",
            .ray = make_ray(origin, direction, 5.0F, infinity, FarShadowVisibility),
            .expected = true,
        },
    };
}

inline void expect_instanced_occlusion(const AccelBackend& backend,
                                       const AccelBackendKind expected_kind) {
    EXPECT_EQ(backend.kind(), expected_kind);
    for (const auto& expectation : instanced_scene_occlusion_expectations()) {
        SCOPED_TRACE(expectation.name);
        const auto result = backend.occluded(expectation.ray);
        ASSERT_TRUE(result.has_value()) << result.error().message;
        EXPECT_EQ(*result, expectation.expected);
    }
}

inline void expect_occlusion_parity(const AccelBackend& analytic, const AccelBackend& embree) {
    for (const auto& expectation : instanced_scene_occlusion_expectations()) {
        SCOPED_TRACE(expectation.name);
        const auto reference = analytic.occluded(expectation.ray);
        const auto accelerated = embree.occluded(expectation.ray);
        ASSERT_TRUE(reference.has_value()) << reference.error().message;
        ASSERT_TRUE(accelerated.has_value()) << accelerated.error().message;
        EXPECT_EQ(*accelerated, *reference);
    }
}

inline void expect_instanced_closest_hits(const AccelBackend& backend,
                                          const AccelBackendKind expected_kind) {
    constexpr auto tolerance = 1.0e-6F;
    EXPECT_EQ(backend.kind(), expected_kind);
    const auto rays = instanced_scene_rays();

    const auto leaf = backend.closest_hit(rays[0]);
    ASSERT_TRUE(leaf.has_value()) << leaf.error().message;
    ASSERT_TRUE(leaf->has_value());
    EXPECT_EQ(leaf->value().object, (renderer::ObjectId{.value = 120U}));
    EXPECT_FLOAT_EQ(leaf->value().triangle.parameter, 4.0F);
    EXPECT_NEAR(leaf->value().triangle.position.x, 9.25F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.position.y, -0.5F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.position.z, 1.0F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.barycentrics.vertex0, 0.5F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.barycentrics.vertex1, 0.25F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.barycentrics.vertex2, 0.25F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.geometric_normal.x, -0.8F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.geometric_normal.y, 0.0F, tolerance);
    EXPECT_NEAR(leaf->value().triangle.geometric_normal.z, -0.6F, tolerance);
    EXPECT_EQ(leaf->value().identifiers, (renderer::SurfaceIdentifiers{
                                             .instance = {.value = 20U},
                                             .geometry = {.value = 84U},
                                             .primitive = {.value = 1U},
                                             .material = {.value = 220U},
                                         }));

    const auto far = backend.closest_hit(rays[1]);
    ASSERT_TRUE(far.has_value()) << far.error().message;
    ASSERT_TRUE(far->has_value());
    EXPECT_EQ(far->value().object, (renderer::ObjectId{.value = 140U}));
    EXPECT_FLOAT_EQ(far->value().triangle.parameter, 8.0F);
    EXPECT_NEAR(far->value().triangle.position.x, 9.25F, tolerance);
    EXPECT_NEAR(far->value().triangle.position.y, -0.5F, tolerance);
    EXPECT_NEAR(far->value().triangle.position.z, 9.0F, tolerance);
    EXPECT_EQ(far->value().identifiers, (renderer::SurfaceIdentifiers{
                                            .instance = {.value = 40U},
                                            .geometry = {.value = 84U},
                                            .primitive = {.value = 1U},
                                            .material = {.value = 240U},
                                        }));
}

inline void expect_closest_hit_parity(const AccelBackend& analytic, const AccelBackend& embree) {
    constexpr auto tolerance = 1.0e-6F;
    for (const auto& ray : instanced_scene_rays()) {
        const auto reference = analytic.closest_hit(ray);
        const auto accelerated = embree.closest_hit(ray);
        ASSERT_TRUE(reference.has_value()) << reference.error().message;
        ASSERT_TRUE(accelerated.has_value()) << accelerated.error().message;
        ASSERT_EQ(reference->has_value(), accelerated->has_value());
        ASSERT_TRUE(reference->has_value());

        EXPECT_EQ(reference->value().object, accelerated->value().object);
        EXPECT_EQ(reference->value().identifiers, accelerated->value().identifiers);
        EXPECT_NEAR(reference->value().triangle.parameter, accelerated->value().triangle.parameter,
                    tolerance);
        EXPECT_NEAR(reference->value().triangle.position.x,
                    accelerated->value().triangle.position.x, tolerance);
        EXPECT_NEAR(reference->value().triangle.position.y,
                    accelerated->value().triangle.position.y, tolerance);
        EXPECT_NEAR(reference->value().triangle.position.z,
                    accelerated->value().triangle.position.z, tolerance);
        EXPECT_NEAR(reference->value().triangle.barycentrics.vertex0,
                    accelerated->value().triangle.barycentrics.vertex0, tolerance);
        EXPECT_NEAR(reference->value().triangle.barycentrics.vertex1,
                    accelerated->value().triangle.barycentrics.vertex1, tolerance);
        EXPECT_NEAR(reference->value().triangle.barycentrics.vertex2,
                    accelerated->value().triangle.barycentrics.vertex2, tolerance);
        EXPECT_NEAR(reference->value().triangle.geometric_normal.x,
                    accelerated->value().triangle.geometric_normal.x, tolerance);
        EXPECT_NEAR(reference->value().triangle.geometric_normal.y,
                    accelerated->value().triangle.geometric_normal.y, tolerance);
        EXPECT_NEAR(reference->value().triangle.geometric_normal.z,
                    accelerated->value().triangle.geometric_normal.z, tolerance);
    }
}

} // namespace blackframe::engine::test
