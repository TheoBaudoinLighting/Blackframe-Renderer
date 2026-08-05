#include <Blackframe/Backends/GPU/CUDA/ClosestHit.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/Shared/SceneBvhAbi.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] testing::AssertionResult select_closest_hit_device() {
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

[[nodiscard]] renderer::Matrix4 translated_matrix(const float z) {
    auto matrix = renderer::identity_matrix<renderer::TransportScalar>();
    matrix(2, 3) = z;
    return matrix;
}

[[nodiscard]] FrameSceneHandle make_closest_hit_scene(const float z = 0.0F) {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.y = 1.0F},
        },
        std::vector(3U, renderer::Normal3{.z = 1.0F}),
        std::vector{
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        std::vector{TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 101U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 201U},
                    .mesh = std::make_shared<const TriangleMesh>(std::move(*mesh)),
                },
            },
        .materials = {SceneMaterial{.id = {.value = 301U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 401U},
                    .parent = std::nullopt,
                    .object = {.value = 101U},
                    .geometry = {.value = 201U},
                    .material = {.value = 301U},
                    .local_to_parent = translated_matrix(z),
                    .visibility_mask = 0x4U,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] FrameSceneHandle make_two_instance_closest_hit_scene(
    const float second_z = 2.0F, const std::array<std::uint32_t, 2> instance_ids = {401U, 402U}) {
    const auto source = make_closest_hit_scene();
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 101U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 201U},
                    .mesh = source->geometries().front().mesh,
                },
            },
        .materials = {SceneMaterial{.id = {.value = 301U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = instance_ids[0]},
                    .parent = std::nullopt,
                    .object = {.value = 101U},
                    .geometry = {.value = 201U},
                    .material = {.value = 301U},
                    .local_to_parent = translated_matrix(0.0F),
                    .visibility_mask = 0x4U,
                },
                SceneInstance{
                    .id = {.value = instance_ids[1]},
                    .parent = std::nullopt,
                    .object = {.value = 101U},
                    .geometry = {.value = 201U},
                    .material = {.value = 301U},
                    .local_to_parent = translated_matrix(second_z),
                    .visibility_mask = 0x4U,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] FrameSceneHandle
make_many_triangle_closest_hit_scene(const std::size_t triangle_count) {
    auto triangles = std::vector<TriangleVertexIndices>(
        triangle_count, TriangleVertexIndices{.vertices = {0U, 1U, 2U}});
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.y = 1.0F},
        },
        std::vector(3U, renderer::Normal3{.z = 1.0F}),
        std::vector{
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        std::move(triangles));
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 101U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 201U},
                    .mesh = std::make_shared<const TriangleMesh>(std::move(*mesh)),
                },
            },
        .materials = {SceneMaterial{.id = {.value = 301U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 401U},
                    .parent = std::nullopt,
                    .object = {.value = 101U},
                    .geometry = {.value = 201U},
                    .material = {.value = 301U},
                    .local_to_parent = translated_matrix(0.0F),
                    .visibility_mask = 0x4U,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

struct UploadedClosestHitScene final {
    CudaSceneSoA scene;
    CudaSceneBvh bvh;
};

[[nodiscard]] UploadedClosestHitScene upload_closest_hit_scene(const FrameScene& scene) {
    auto uploaded = CudaSceneSoA::upload(scene);
    if (!uploaded) {
        throw std::runtime_error{uploaded.error().message};
    }
    auto bvh = CudaSceneBvh::build(*uploaded);
    if (!bvh) {
        throw std::runtime_error{bvh.error().message};
    }
    return UploadedClosestHitScene{
        .scene = std::move(*uploaded),
        .bvh = std::move(*bvh),
    };
}

[[nodiscard]] renderer::Ray make_closest_hit_ray(const renderer::RayMask mask = 0x4U,
                                                 const float time = 0.0F) {
    auto ray = renderer::Ray::create(
        renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F}, renderer::Vector3{.z = 1.0F}, 0.0F,
        std::numeric_limits<float>::infinity(), time, mask, renderer::VacuumMedium);
    if (!ray) {
        throw std::runtime_error{ray.error().message};
    }
    return *ray;
}

TEST(CudaClosestHit, TracesOneTriangleAndPreservesTheFrozenHitContract) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto frame = make_closest_hit_scene();
    auto device = upload_closest_hit_scene(*frame);

    const auto hit = trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray());
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    ASSERT_TRUE(hit->has_value());
    EXPECT_EQ(hit->value().object, (renderer::ObjectId{.value = 101U}));
    EXPECT_EQ(hit->value().identifiers, (renderer::SurfaceIdentifiers{
                                            .instance = {.value = 401U},
                                            .geometry = {.value = 201U},
                                            .primitive = {.value = 0U},
                                            .material = {.value = 301U},
                                        }));
    EXPECT_FLOAT_EQ(hit->value().triangle.parameter, 1.0F);
    EXPECT_NEAR(hit->value().triangle.barycentrics.vertex0, 0.5F, 1.0e-6F);
    EXPECT_NEAR(hit->value().triangle.barycentrics.vertex1, 0.25F, 1.0e-6F);
    EXPECT_NEAR(hit->value().triangle.barycentrics.vertex2, 0.25F, 1.0e-6F);
    EXPECT_NEAR(hit->value().triangle.geometric_normal.x, 0.0F, 1.0e-6F);
    EXPECT_NEAR(hit->value().triangle.geometric_normal.y, 0.0F, 1.0e-6F);
    EXPECT_NEAR(hit->value().triangle.geometric_normal.z, 1.0F, 1.0e-6F);

    const auto masked =
        trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray(0x8U));
    ASSERT_TRUE(masked.has_value()) << masked.error().message;
    EXPECT_FALSE(masked->has_value());
}

TEST(CudaClosestHit, RepresentsEmptyWorkAndEmptyScenesWithoutSyntheticHits) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto frame = make_closest_hit_scene();
    auto device = upload_closest_hit_scene(*frame);
    const auto empty_batch = trace_cuda_closest_hits(device.scene, device.bvh, {});
    ASSERT_TRUE(empty_batch.has_value()) << empty_batch.error().message;
    EXPECT_TRUE(empty_batch->empty());

    const auto empty_frame = FrameScene::create(FrameSceneDescription{});
    ASSERT_TRUE(empty_frame.has_value()) << empty_frame.error().message;
    auto empty_device = upload_closest_hit_scene(**empty_frame);
    const auto miss =
        trace_cuda_closest_hit(empty_device.scene, empty_device.bvh, make_closest_hit_ray());
    ASSERT_TRUE(miss.has_value()) << miss.error().message;
    EXPECT_FALSE(miss->has_value());
}

TEST(CudaClosestHit, RejectsMismatchedScenesInvalidTimeAndInsufficientQueryMemory) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto first_frame = make_closest_hit_scene();
    const auto second_frame = make_closest_hit_scene(2.0F);
    auto first = upload_closest_hit_scene(*first_frame);
    auto second = upload_closest_hit_scene(*second_frame);
    const auto ray = make_closest_hit_ray();

    const auto mismatch = trace_cuda_closest_hit(second.scene, first.bvh, ray);
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(mismatch.error().message.empty());

    const auto invalid_time =
        trace_cuda_closest_hit(first.scene, first.bvh, make_closest_hit_ray(0x4U, 1.25F));
    ASSERT_FALSE(invalid_time.has_value());
    EXPECT_EQ(invalid_time.error().code, core::StatusCode::invalid_argument);

    constexpr auto required_query_bytes =
        sizeof(xpu::shared::TransportRay) + sizeof(xpu::shared::SceneClosestHitResult);
    const auto exhausted = trace_cuda_closest_hit(
        first.scene, first.bvh, ray,
        CudaClosestHitQueryOptions{
            .device_memory_budget =
                xpu::cuda::DeviceMemoryBudget{.maximum_bytes = required_query_bytes - 1U},
        });
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error().code, core::StatusCode::resource_exhausted);
    EXPECT_NE(exhausted.error().message.find("explicit device-memory budget"), std::string::npos);
}

TEST(CudaClosestHit, LowLevelLauncherRejectsMissingStorageExplicitly) {
    ASSERT_TRUE(select_closest_hit_device());
    EXPECT_EQ(
        blackframe_cuda_launch_scene_closest_hit(nullptr, 0U, nullptr, 0U, nullptr, 1U, nullptr),
        static_cast<int>(cudaErrorInvalidValue));
}

TEST(CudaClosestHit, ReportsCyclicDeviceTopologyInsteadOfLooping) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto frame = make_two_instance_closest_hit_scene();
    auto device = upload_closest_hit_scene(*frame);
    auto bytes = std::vector<std::uint8_t>(device.bvh.size_bytes());
    auto status =
        cudaMemcpy(bytes.data(), device.bvh.device_data(), bytes.size(), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

    auto header = xpu::shared::SceneBvhHeader{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    ASSERT_EQ(header.tlas_node_count, 3U);
    ASSERT_EQ(header.tlas_root_node, 0U);
    const auto root_offset = static_cast<std::size_t>(
        header.arrays[xpu::shared::scene_bvh_array::tlas_node].offset_bytes);
    auto root = xpu::shared::SceneBvhNode{};
    std::memcpy(&root, bytes.data() + root_offset, sizeof(root));
    ASSERT_EQ(root.kind, static_cast<std::uint32_t>(xpu::shared::SceneBvhNodeKind::internal));
    root.first_child = header.tlas_root_node;
    std::memcpy(bytes.data() + root_offset, &root, sizeof(root));
    status = cudaMemcpy(const_cast<std::uint8_t*>(device.bvh.device_data()), bytes.data(),
                        bytes.size(), cudaMemcpyHostToDevice);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

    const auto hit = trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray());
    ASSERT_FALSE(hit.has_value());
    EXPECT_EQ(hit.error().code, core::StatusCode::internal_error);
    EXPECT_NE(hit.error().message.find("invalid BVH topology"), std::string::npos);
}

TEST(CudaClosestHit, RejectsNonCanonicalReservedBvhRecords) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto frame = make_closest_hit_scene();

    {
        auto device = upload_closest_hit_scene(*frame);
        auto bytes = std::vector<std::uint8_t>(device.bvh.size_bytes());
        auto status = cudaMemcpy(bytes.data(), device.bvh.device_data(), bytes.size(),
                                 cudaMemcpyDeviceToHost);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        auto header = xpu::shared::SceneBvhHeader{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        const auto offset = static_cast<std::size_t>(
            header.arrays[xpu::shared::scene_bvh_array::blas].offset_bytes);
        auto record = xpu::shared::SceneBvhBlas{};
        std::memcpy(&record, bytes.data() + offset, sizeof(record));
        record.reserved[0] = 1U;
        std::memcpy(bytes.data() + offset, &record, sizeof(record));
        status = cudaMemcpy(const_cast<std::uint8_t*>(device.bvh.device_data()), bytes.data(),
                            bytes.size(), cudaMemcpyHostToDevice);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

        const auto hit = trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray());
        ASSERT_FALSE(hit.has_value());
        EXPECT_EQ(hit.error().code, core::StatusCode::internal_error);
        EXPECT_NE(hit.error().message.find("invalid BVH topology"), std::string::npos);
    }

    {
        auto device = upload_closest_hit_scene(*frame);
        auto bytes = std::vector<std::uint8_t>(device.bvh.size_bytes());
        auto status = cudaMemcpy(bytes.data(), device.bvh.device_data(), bytes.size(),
                                 cudaMemcpyDeviceToHost);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        auto header = xpu::shared::SceneBvhHeader{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        const auto offset = static_cast<std::size_t>(
            header.arrays[xpu::shared::scene_bvh_array::instance_reference].offset_bytes);
        auto record = xpu::shared::SceneBvhInstanceReference{};
        std::memcpy(&record, bytes.data() + offset, sizeof(record));
        record.reserved[1] = 1U;
        std::memcpy(bytes.data() + offset, &record, sizeof(record));
        status = cudaMemcpy(const_cast<std::uint8_t*>(device.bvh.device_data()), bytes.data(),
                            bytes.size(), cudaMemcpyHostToDevice);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

        const auto hit = trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray());
        ASSERT_FALSE(hit.has_value());
        EXPECT_EQ(hit.error().code, core::StatusCode::internal_error);
        EXPECT_NE(hit.error().message.find("invalid BVH topology"), std::string::npos);
    }
}

TEST(CudaClosestHit, PreservesLaneOrderAcrossBlocksAndTraversesADeepBlas) {
    ASSERT_TRUE(select_closest_hit_device());
    constexpr auto triangle_count = std::size_t{257U};
    constexpr auto ray_count = std::size_t{513U};
    const auto frame = make_many_triangle_closest_hit_scene(triangle_count);
    auto device = upload_closest_hit_scene(*frame);

    auto rays = std::vector<renderer::Ray>{};
    rays.reserve(ray_count);
    for (auto index = std::size_t{}; index < ray_count; ++index) {
        const auto signed_zero = (index % 4U) < 2U ? 0.0F : -0.0F;
        auto ray = renderer::Ray::create(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                         renderer::Vector3{.x = signed_zero, .z = 1.0F}, 0.0F,
                                         std::numeric_limits<float>::infinity(), 0.0F,
                                         index % 2U == 0U ? renderer::RayMask{0x4U}
                                                          : renderer::RayMask{0x8U},
                                         renderer::VacuumMedium);
        ASSERT_TRUE(ray.has_value()) << ray.error().message;
        rays.push_back(*ray);
    }

    const auto hits = trace_cuda_closest_hits(device.scene, device.bvh, rays);
    ASSERT_TRUE(hits.has_value()) << hits.error().message;
    ASSERT_EQ(hits->size(), ray_count);
    for (auto index = std::size_t{}; index < ray_count; ++index) {
        SCOPED_TRACE(index);
        if (index % 2U != 0U) {
            EXPECT_FALSE((*hits)[index].has_value());
            continue;
        }
        ASSERT_TRUE((*hits)[index].has_value());
        EXPECT_EQ((*hits)[index]->identifiers.primitive, (renderer::PrimitiveId{.value = 0U}));
        EXPECT_FLOAT_EQ((*hits)[index]->triangle.parameter, 1.0F);
    }
}

TEST(CudaClosestHit, ResolvesCoincidentInstanceTiesByStableIdentifier) {
    ASSERT_TRUE(select_closest_hit_device());
    const auto frame = make_two_instance_closest_hit_scene(0.0F, {402U, 401U});
    auto device = upload_closest_hit_scene(*frame);

    const auto hit = trace_cuda_closest_hit(device.scene, device.bvh, make_closest_hit_ray());
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    ASSERT_TRUE(hit->has_value());
    EXPECT_EQ(hit->value().identifiers.instance, (renderer::InstanceId{.value = 401U}));
    EXPECT_EQ(hit->value().identifiers.primitive, (renderer::PrimitiveId{.value = 0U}));
    EXPECT_FLOAT_EQ(hit->value().triangle.parameter, 1.0F);
}

} // namespace
} // namespace blackframe::engine
