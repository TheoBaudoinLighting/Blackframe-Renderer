#include "../../../Engine/AccelBackendContract.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/GPU/CUDA/ClosestHit.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Engine/TriangleMeshImport.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr auto HitTolerance = renderer::TransportScalar{1.0e-6F};
constexpr auto MeshParityExtent = renderer::RenderExtent{.width = 8U, .height = 8U};

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

struct DeviceScene final {
    CudaSceneSoA scene;
    CudaSceneBvh bvh;
};

[[nodiscard]] DeviceScene upload_and_build(const FrameScene& scene) {
    auto uploaded = CudaSceneSoA::upload(scene);
    if (!uploaded) {
        throw std::runtime_error{uploaded.error().message};
    }
    auto bvh = CudaSceneBvh::build(*uploaded);
    if (!bvh) {
        throw std::runtime_error{bvh.error().message};
    }
    return DeviceScene{
        .scene = std::move(*uploaded),
        .bvh = std::move(*bvh),
    };
}

void expect_point_near(const renderer::Point3 actual, const renderer::Point3 expected) {
    EXPECT_NEAR(actual.x, expected.x, HitTolerance);
    EXPECT_NEAR(actual.y, expected.y, HitTolerance);
    EXPECT_NEAR(actual.z, expected.z, HitTolerance);
}

void expect_normal_near(const renderer::Normal3 actual, const renderer::Normal3 expected) {
    EXPECT_NEAR(actual.x, expected.x, HitTolerance);
    EXPECT_NEAR(actual.y, expected.y, HitTolerance);
    EXPECT_NEAR(actual.z, expected.z, HitTolerance);
}

void expect_barycentrics_near(const renderer::TriangleBarycentrics actual,
                              const renderer::TriangleBarycentrics expected) {
    EXPECT_NEAR(actual.vertex0, expected.vertex0, HitTolerance);
    EXPECT_NEAR(actual.vertex1, expected.vertex1, HitTolerance);
    EXPECT_NEAR(actual.vertex2, expected.vertex2, HitTolerance);
}

void expect_barycentrics_on_triangle(const renderer::TriangleBarycentrics barycentrics,
                                     const bool require_boundary) {
    const auto values =
        std::array{barycentrics.vertex0, barycentrics.vertex1, barycentrics.vertex2};
    for (const auto value : values) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, -HitTolerance);
        EXPECT_LE(value, renderer::TransportScalar{1} + HitTolerance);
    }
    EXPECT_NEAR(barycentrics.vertex0 + barycentrics.vertex1 + barycentrics.vertex2, 1.0F,
                HitTolerance);
    if (require_boundary) {
        const auto minimum =
            std::min({std::abs(values[0]), std::abs(values[1]), std::abs(values[2])});
        EXPECT_LE(minimum, HitTolerance);
    }
}

void expect_strict_hit_parity(const std::optional<AccelHit>& cuda_hit,
                              const std::optional<AccelHit>& embree_hit) {
    ASSERT_EQ(cuda_hit.has_value(), embree_hit.has_value());
    if (!embree_hit) {
        return;
    }

    ASSERT_TRUE(cuda_hit.has_value());
    EXPECT_EQ(cuda_hit->object, embree_hit->object);
    EXPECT_EQ(cuda_hit->identifiers, embree_hit->identifiers);
    EXPECT_NEAR(cuda_hit->triangle.parameter, embree_hit->triangle.parameter, HitTolerance);
    expect_point_near(cuda_hit->triangle.position, embree_hit->triangle.position);
    expect_barycentrics_near(cuda_hit->triangle.barycentrics, embree_hit->triangle.barycentrics);
    expect_normal_near(cuda_hit->triangle.geometric_normal, embree_hit->triangle.geometric_normal);
}

void expect_shared_edge_hit_parity(const std::optional<AccelHit>& cuda_hit,
                                   const std::optional<AccelHit>& embree_hit) {
    ASSERT_TRUE(cuda_hit.has_value());
    ASSERT_TRUE(embree_hit.has_value());

    EXPECT_EQ(cuda_hit->object, embree_hit->object);
    EXPECT_EQ(cuda_hit->identifiers.instance, embree_hit->identifiers.instance);
    EXPECT_EQ(cuda_hit->identifiers.geometry, embree_hit->identifiers.geometry);
    EXPECT_EQ(cuda_hit->identifiers.material, embree_hit->identifiers.material);
    EXPECT_LT(cuda_hit->identifiers.primitive.value, 2U);
    EXPECT_LT(embree_hit->identifiers.primitive.value, 2U);
    EXPECT_NEAR(cuda_hit->triangle.parameter, embree_hit->triangle.parameter, HitTolerance);
    expect_point_near(cuda_hit->triangle.position, embree_hit->triangle.position);
    expect_normal_near(cuda_hit->triangle.geometric_normal, embree_hit->triangle.geometric_normal);
    expect_barycentrics_on_triangle(cuda_hit->triangle.barycentrics, true);
    expect_barycentrics_on_triangle(embree_hit->triangle.barycentrics, true);
    if (cuda_hit->identifiers.primitive == embree_hit->identifiers.primitive) {
        expect_barycentrics_near(cuda_hit->triangle.barycentrics,
                                 embree_hit->triangle.barycentrics);
    }
}

[[nodiscard]] std::vector<std::optional<AccelHit>>
trace_embree_hits(const AccelBackend& embree, const std::span<const renderer::Ray> rays) {
    auto hits = std::vector<std::optional<AccelHit>>{};
    hits.reserve(rays.size());
    for (const auto& ray : rays) {
        auto hit = embree.closest_hit(ray);
        if (!hit) {
            throw std::runtime_error{hit.error().message};
        }
        hits.push_back(std::move(*hit));
    }
    return hits;
}

void expect_strict_batch_parity(const DeviceScene& device_scene, const AccelBackend& embree,
                                const std::span<const renderer::Ray> rays) {
    auto cuda_hits = trace_cuda_closest_hits(device_scene.scene, device_scene.bvh, rays);
    ASSERT_TRUE(cuda_hits.has_value()) << cuda_hits.error().message;
    ASSERT_EQ(cuda_hits->size(), rays.size());
    const auto embree_hits = trace_embree_hits(embree, rays);

    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        expect_strict_hit_parity((*cuda_hits)[index], embree_hits[index]);
    }
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_shared_edge_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{},
            renderer::Point3{.x = 2.0F},
            renderer::Point3{.y = 2.0F},
            renderer::Point3{.x = 2.0F, .y = 2.0F},
        },
        std::vector(4U, renderer::Normal3{.z = 1.0F}),
        std::vector{
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
        },
        std::vector{
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {3U, 2U, 1U}},
        });
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_translated_oblique_shared_edge_mesh() {
    constexpr auto base = renderer::Point3{
        .x = 1'048'576.0F,
        .y = -1'048'576.0F,
        .z = 1'048'576.0F,
    };
    constexpr auto tangent0 = renderer::Vector3{.x = 16.0F, .z = 4.0F};
    constexpr auto tangent1 = renderer::Vector3{.y = 16.0F, .z = 8.0F};
    const auto vertex1 = base + tangent0;
    const auto vertex2 = base + tangent1;
    const auto opposite = base + tangent0 + tangent1;
    auto mesh = TriangleMesh::create(std::vector{base, vertex1, vertex2, opposite},
                                     std::vector(4U, renderer::Normal3{.z = 1.0F}),
                                     std::vector{
                                         renderer::Point2{},
                                         renderer::Point2{.x = 1.0F},
                                         renderer::Point2{.y = 1.0F},
                                         renderer::Point2{.x = 1.0F, .y = 1.0F},
                                     },
                                     std::vector{
                                         TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
                                         TriangleVertexIndices{.vertices = {3U, 2U, 1U}},
                                     });
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] FrameSceneHandle make_shared_edge_scene() {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 501U}}},
        .geometries = {SceneGeometry{.id = {.value = 601U}, .mesh = make_shared_edge_mesh()}},
        .materials = {SceneMaterial{.id = {.value = 701U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 801U},
                    .parent = std::nullopt,
                    .object = {.value = 501U},
                    .geometry = {.value = 601U},
                    .material = {.value = 701U},
                    .local_to_parent = test::identity_transform_matrix(),
                    .visibility_mask = renderer::AllRayVisibility,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] FrameSceneHandle make_translated_oblique_shared_edge_scene() {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 511U}}},
        .geometries = {SceneGeometry{.id = {.value = 611U},
                                     .mesh = make_translated_oblique_shared_edge_mesh()}},
        .materials = {SceneMaterial{.id = {.value = 711U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 811U},
                    .parent = std::nullopt,
                    .object = {.value = 511U},
                    .geometry = {.value = 611U},
                    .material = {.value = 711U},
                    .local_to_parent = test::identity_transform_matrix(),
                    .visibility_mask = renderer::AllRayVisibility,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] FrameSceneHandle make_imported_mesh_scene(TriangleMesh mesh) {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 901U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 902U},
                    .mesh = std::make_shared<const TriangleMesh>(std::move(mesh)),
                },
            },
        .materials = {SceneMaterial{.id = {.value = 903U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 904U},
                    .parent = std::nullopt,
                    .object = {.value = 901U},
                    .geometry = {.value = 902U},
                    .material = {.value = 903U},
                    .local_to_parent = test::identity_transform_matrix(),
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] std::vector<renderer::Ray> mesh_parity_camera_rays() {
    const auto frame = renderer::OrthonormalFrame::from_normal_and_tangent(
        renderer::Normal3{.z = 1.0F}, renderer::Vector3{.x = 1.0F});
    if (!frame) {
        throw std::runtime_error{frame.error().message};
    }
    const auto camera =
        renderer::PinholeCamera::create(renderer::Point3{}, *frame, MeshParityExtent,
                                        std::numbers::pi_v<renderer::TransportScalar> / 2.0F, 0.0F,
                                        4.0F, renderer::AllRayVisibility, renderer::VacuumMedium);
    if (!camera) {
        throw std::runtime_error{camera.error().message};
    }

    auto rays = std::vector<renderer::Ray>{};
    rays.reserve(static_cast<std::size_t>(MeshParityExtent.width) * MeshParityExtent.height);
    for (auto y = std::uint32_t{}; y < MeshParityExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < MeshParityExtent.width; ++x) {
            auto ray = camera->generate_primary_ray(
                renderer::PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = 0U,
                    .seed = 0U,
                },
                renderer::PixelJitterMode::center, 0.25F);
            if (!ray) {
                throw std::runtime_error{ray.error().message};
            }
            rays.push_back(*ray);
        }
    }
    return rays;
}

TEST(CudaClosestHitParity, MatchesEmbreeForMasksClippingMissesAndCoplanarGeometry) {
    ASSERT_TRUE(select_test_device());
    const auto scene = test::make_basic_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    const auto infinity = std::numeric_limits<renderer::TransportScalar>::infinity();
    const auto origin = renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F};
    const auto direction = renderer::Vector3{.z = 1.0F};
    const auto rays = std::vector{
        test::make_ray(origin, direction),
        test::make_ray(origin, direction, 0.0F, infinity, test::FrontVisibility),
        test::make_ray(origin, direction, 0.0F, infinity, test::BackVisibility),
        test::make_ray(renderer::Point3{.x = 1.25F, .y = 1.25F, .z = -1.0F}, direction),
        test::make_ray(origin, direction, 0.0F, 0.75F),
        test::make_ray(origin, direction, 1.0F, 1.0F, test::FrontVisibility),
        test::make_ray(origin, direction, 0.0F, std::nextafter(1.0F, 0.0F), test::FrontVisibility),
        test::make_ray(origin, direction, std::nextafter(1.0F, 2.0F), infinity,
                       test::FrontVisibility),
        test::make_ray(origin, direction, 0.0F, infinity, renderer::RayMask{1U << 8U}),
        test::make_ray(origin, direction, 0.0F, infinity, renderer::RayMask{}),
    };

    expect_strict_batch_parity(device_scene, **embree, rays);

    const auto cuda_single = trace_cuda_closest_hit(device_scene.scene, device_scene.bvh, rays[0]);
    ASSERT_TRUE(cuda_single.has_value()) << cuda_single.error().message;
    const auto embree_single = (*embree)->closest_hit(rays[0]);
    ASSERT_TRUE(embree_single.has_value()) << embree_single.error().message;
    expect_strict_hit_parity(*cuda_single, *embree_single);
}

TEST(CudaClosestHitParity, MatchesEmbreeForHierarchicalMirroredInstances) {
    ASSERT_TRUE(select_test_device());
    const auto scene = test::make_instanced_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    const auto base_rays = test::instanced_scene_rays();
    const auto infinity = std::numeric_limits<renderer::TransportScalar>::infinity();
    const auto origin = renderer::Point3{.x = 9.25F, .y = -0.5F, .z = -7.0F};
    const auto direction = renderer::Vector3{.z = 2.0F};
    const auto unrelated_visibility = renderer::RayMask{1U << 8U};
    const auto rays = std::vector{
        base_rays[0],
        base_rays[1],
        test::make_ray(origin, direction, 0.0F, infinity, test::NearShadowVisibility),
        test::make_ray(origin, direction, 0.0F, infinity, test::FarShadowVisibility),
        test::make_ray(origin, direction, 0.0F, infinity,
                       test::NearShadowVisibility | test::FarShadowVisibility),
        test::make_ray(origin, direction, 0.0F, infinity, unrelated_visibility),
        test::make_ray(origin, direction, 0.0F, infinity, renderer::RayMask{}),
        test::make_ray(origin, direction, 0.0F, 6.0F, test::NearShadowVisibility),
        test::make_ray(origin, direction, 0.0F, 6.0F, test::FarShadowVisibility),
        test::make_ray(origin, direction, 5.0F, infinity, test::NearShadowVisibility),
        test::make_ray(origin, direction, 5.0F, infinity, test::FarShadowVisibility),
    };

    expect_strict_batch_parity(device_scene, **embree, rays);
}

TEST(CudaClosestHitParity, KeepsSharedEdgesWatertightAgainstEmbree) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_shared_edge_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    constexpr auto parameters = std::array{0.0F, 0.125F, 0.25F, 0.5F, 0.75F, 0.875F, 1.0F};
    auto edge_rays = std::vector<renderer::Ray>{};
    edge_rays.reserve(parameters.size() * 2U);
    for (const auto parameter : parameters) {
        const auto point = renderer::Point3{
            .x = 2.0F * (1.0F - parameter),
            .y = 2.0F * parameter,
        };
        edge_rays.push_back(
            test::make_ray(point + renderer::Vector3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}));
        edge_rays.push_back(
            test::make_ray(point + renderer::Vector3{.z = -1.0F}, renderer::Vector3{.z = 1.0F}));
    }

    auto cuda_edge_hits = trace_cuda_closest_hits(device_scene.scene, device_scene.bvh, edge_rays);
    ASSERT_TRUE(cuda_edge_hits.has_value()) << cuda_edge_hits.error().message;
    ASSERT_EQ(cuda_edge_hits->size(), edge_rays.size());
    const auto embree_edge_hits = trace_embree_hits(**embree, edge_rays);
    for (auto index = std::size_t{}; index < edge_rays.size(); ++index) {
        SCOPED_TRACE(index);
        expect_shared_edge_hit_parity((*cuda_edge_hits)[index], embree_edge_hits[index]);
    }

    const auto side_rays = std::array{
        test::make_ray(renderer::Point3{.x = 1.0F, .y = std::nextafter(1.0F, 0.0F), .z = 1.0F},
                       renderer::Vector3{.z = -1.0F}),
        test::make_ray(renderer::Point3{.x = 1.0F, .y = std::nextafter(1.0F, 2.0F), .z = 1.0F},
                       renderer::Vector3{.z = -1.0F}),
    };
    auto cuda_side_hits = trace_cuda_closest_hits(device_scene.scene, device_scene.bvh, side_rays);
    ASSERT_TRUE(cuda_side_hits.has_value()) << cuda_side_hits.error().message;
    ASSERT_EQ(cuda_side_hits->size(), side_rays.size());
    const auto embree_side_hits = trace_embree_hits(**embree, side_rays);
    for (auto index = std::size_t{}; index < side_rays.size(); ++index) {
        SCOPED_TRACE(index);
        ASSERT_TRUE((*cuda_side_hits)[index].has_value());
        ASSERT_TRUE(embree_side_hits[index].has_value());
        // At one ULP from the shared edge, Embree may still report either
        // closed-edge owner. CUDA must nevertheless choose the geometric side
        // deterministically and may not turn either ray into a crack.
        expect_shared_edge_hit_parity((*cuda_side_hits)[index], embree_side_hits[index]);
        EXPECT_EQ((*cuda_side_hits)[index]->identifiers.primitive.value,
                  static_cast<std::uint32_t>(index));
    }
}

TEST(CudaClosestHitParity, PreservesTranslatedObliqueSharedEdgesAgainstEmbree) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_translated_oblique_shared_edge_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    constexpr auto base = renderer::Point3{
        .x = 1'048'576.0F,
        .y = -1'048'576.0F,
        .z = 1'048'576.0F,
    };
    constexpr auto tangent0 = renderer::Vector3{.x = 16.0F, .z = 4.0F};
    constexpr auto tangent1 = renderer::Vector3{.y = 16.0F, .z = 8.0F};
    const auto vertex1 = base + tangent0;
    const auto vertex2 = base + tangent1;
    constexpr auto edge_parameters = std::array{0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    constexpr auto directions = std::array{
        renderer::Vector3{.x = 1.0F, .y = 2.0F, .z = -4.0F},
        renderer::Vector3{.x = -3.0F, .y = 2.0F, .z = 5.0F},
        renderer::Vector3{.x = 8.0F, .y = 1.0F, .z = -1.0F},
        renderer::Vector3{.x = 1.0F, .y = 8.0F, .z = -1.0F},
    };
    auto rays = std::vector<renderer::Ray>{};
    rays.reserve(edge_parameters.size() * directions.size());
    for (const auto edge_parameter : edge_parameters) {
        const auto point = vertex1 + (vertex2 - vertex1) * edge_parameter;
        for (const auto direction : directions) {
            rays.push_back(test::make_ray(point - direction * 8.0F, direction));
        }
    }
    const auto clipped_point = vertex2;
    const auto clipped_direction = directions[1];
    rays.push_back(
        test::make_ray(clipped_point - clipped_direction * 8.0F, clipped_direction, 8.0F, 8.0F));

    auto cuda_hits = trace_cuda_closest_hits(device_scene.scene, device_scene.bvh, rays);
    ASSERT_TRUE(cuda_hits.has_value()) << cuda_hits.error().message;
    ASSERT_EQ(cuda_hits->size(), rays.size());
    const auto embree_hits = trace_embree_hits(**embree, rays);
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        expect_shared_edge_hit_parity((*cuda_hits)[index], embree_hits[index]);
        ASSERT_TRUE((*cuda_hits)[index].has_value());
        EXPECT_FLOAT_EQ((*cuda_hits)[index]->triangle.parameter, 8.0F);
    }
}

TEST(CudaClosestHitParity, MatchesEmbreeForTheReferenceObjAndPlyMeshCorpus) {
    ASSERT_TRUE(select_test_device());
    const auto fixture_directory = std::filesystem::path{BLACKFRAME_CUDA_MESH_PARITY_FIXTURE_DIR};
    auto obj_mesh = load_obj_triangle_mesh(fixture_directory / "reference-mesh.obj");
    auto ply_mesh = load_ply_triangle_mesh(fixture_directory / "reference-mesh.ply");
    ASSERT_TRUE(obj_mesh.has_value()) << obj_mesh.error().message;
    ASSERT_TRUE(ply_mesh.has_value()) << ply_mesh.error().message;
    auto obj_scene = make_imported_mesh_scene(std::move(*obj_mesh));
    auto ply_scene = make_imported_mesh_scene(std::move(*ply_mesh));
    auto obj_device = upload_and_build(*obj_scene);
    auto ply_device = upload_and_build(*ply_scene);
    auto obj_embree = create_embree_accel_backend(obj_scene);
    auto ply_embree = create_embree_accel_backend(ply_scene);
    ASSERT_TRUE(obj_embree.has_value()) << obj_embree.error().message;
    ASSERT_TRUE(ply_embree.has_value()) << ply_embree.error().message;
    const auto rays = mesh_parity_camera_rays();

    auto obj_cuda_hits = trace_cuda_closest_hits(obj_device.scene, obj_device.bvh, rays);
    auto ply_cuda_hits = trace_cuda_closest_hits(ply_device.scene, ply_device.bvh, rays);
    ASSERT_TRUE(obj_cuda_hits.has_value()) << obj_cuda_hits.error().message;
    ASSERT_TRUE(ply_cuda_hits.has_value()) << ply_cuda_hits.error().message;
    ASSERT_EQ(obj_cuda_hits->size(), rays.size());
    ASSERT_EQ(ply_cuda_hits->size(), rays.size());
    const auto obj_embree_hits = trace_embree_hits(**obj_embree, rays);
    const auto ply_embree_hits = trace_embree_hits(**ply_embree, rays);

    auto obj_hit_count = std::size_t{};
    auto ply_hit_count = std::size_t{};
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        expect_strict_hit_parity((*obj_cuda_hits)[index], obj_embree_hits[index]);
        expect_strict_hit_parity((*ply_cuda_hits)[index], ply_embree_hits[index]);
        obj_hit_count += (*obj_cuda_hits)[index].has_value() ? 1U : 0U;
        ply_hit_count += (*ply_cuda_hits)[index].has_value() ? 1U : 0U;
    }
    EXPECT_EQ(obj_hit_count, 16U);
    EXPECT_EQ(ply_hit_count, 16U);
}

} // namespace
} // namespace blackframe::engine
