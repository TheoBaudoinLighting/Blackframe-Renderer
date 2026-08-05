#include "../../../Engine/AccelBackendContract.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/GPU/CUDA/Occlusion.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Engine/TriangleMeshImport.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/Ray.hpp>
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
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

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

[[nodiscard]] std::vector<bool> trace_embree_occlusions(const AccelBackend& embree,
                                                        const std::span<const renderer::Ray> rays) {
    auto occlusions = std::vector<bool>{};
    occlusions.reserve(rays.size());
    for (const auto& ray : rays) {
        auto occluded = embree.occluded(ray);
        if (!occluded) {
            throw std::runtime_error{occluded.error().message};
        }
        occlusions.push_back(*occluded);
    }
    return occlusions;
}

void expect_batch_parity(const DeviceScene& device_scene, const AccelBackend& embree,
                         const std::span<const renderer::Ray> rays,
                         const std::span<const bool> expected = {}) {
    auto cuda_occlusions = trace_cuda_occlusions(device_scene.scene, device_scene.bvh, rays);
    ASSERT_TRUE(cuda_occlusions.has_value()) << cuda_occlusions.error().message;
    ASSERT_EQ(cuda_occlusions->size(), rays.size());
    const auto embree_occlusions = trace_embree_occlusions(embree, rays);
    ASSERT_EQ(embree_occlusions.size(), rays.size());
    if (!expected.empty()) {
        ASSERT_EQ(expected.size(), rays.size());
    }

    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_EQ((*cuda_occlusions)[index], embree_occlusions[index]);
        if (!expected.empty()) {
            EXPECT_EQ((*cuda_occlusions)[index], expected[index]);
        }
    }
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

TEST(CudaOcclusionParity, MatchesEmbreeForShadowSegmentsMasksClippingAndMisses) {
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
    constexpr auto expected =
        std::array{true, true, true, false, false, true, false, false, false, false};

    expect_batch_parity(device_scene, **embree, rays, expected);

    const auto cuda_hit = trace_cuda_occlusion(device_scene.scene, device_scene.bvh, rays.front());
    ASSERT_TRUE(cuda_hit.has_value()) << cuda_hit.error().message;
    const auto embree_hit = (*embree)->occluded(rays.front());
    ASSERT_TRUE(embree_hit.has_value()) << embree_hit.error().message;
    EXPECT_EQ(*cuda_hit, *embree_hit);
    EXPECT_TRUE(*cuda_hit);

    const auto cuda_miss = trace_cuda_occlusion(device_scene.scene, device_scene.bvh, rays.back());
    ASSERT_TRUE(cuda_miss.has_value()) << cuda_miss.error().message;
    const auto embree_miss = (*embree)->occluded(rays.back());
    ASSERT_TRUE(embree_miss.has_value()) << embree_miss.error().message;
    EXPECT_EQ(*cuda_miss, *embree_miss);
    EXPECT_FALSE(*cuda_miss);
}

TEST(CudaOcclusionParity, MatchesEmbreeForHierarchicalMirroredInstances) {
    ASSERT_TRUE(select_test_device());
    const auto scene = test::make_instanced_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    const auto expectations = test::instanced_scene_occlusion_expectations();
    auto rays = std::vector<renderer::Ray>{};
    auto expected = std::vector<bool>{};
    rays.reserve(expectations.size());
    expected.reserve(expectations.size());
    for (const auto& expectation : expectations) {
        rays.push_back(expectation.ray);
        expected.push_back(expectation.expected);
    }

    auto cuda_occlusions = trace_cuda_occlusions(device_scene.scene, device_scene.bvh, rays);
    ASSERT_TRUE(cuda_occlusions.has_value()) << cuda_occlusions.error().message;
    ASSERT_EQ(cuda_occlusions->size(), rays.size());
    const auto embree_occlusions = trace_embree_occlusions(**embree, rays);
    for (auto index = std::size_t{}; index < expectations.size(); ++index) {
        SCOPED_TRACE(std::string_view{expectations[index].name});
        EXPECT_EQ((*cuda_occlusions)[index], embree_occlusions[index]);
        EXPECT_EQ((*cuda_occlusions)[index], expected[index]);
    }
}

TEST(CudaOcclusionParity, PreservesLaneOrderAcrossMultipleKernelBlocks) {
    ASSERT_TRUE(select_test_device());
    const auto scene = test::make_instanced_scene();
    auto device_scene = upload_and_build(*scene);
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    const auto patterns = test::instanced_scene_occlusion_expectations();
    constexpr auto RayCount = std::size_t{1'025U};
    auto rays = std::vector<renderer::Ray>{};
    auto expected = std::vector<bool>{};
    rays.reserve(RayCount);
    expected.reserve(RayCount);
    for (auto index = std::size_t{}; index < RayCount; ++index) {
        const auto& pattern = patterns[index % patterns.size()];
        rays.push_back(pattern.ray);
        expected.push_back(pattern.expected);
    }

    auto cuda_occlusions = trace_cuda_occlusions(device_scene.scene, device_scene.bvh, rays);
    ASSERT_TRUE(cuda_occlusions.has_value()) << cuda_occlusions.error().message;
    ASSERT_EQ(cuda_occlusions->size(), rays.size());
    const auto embree_occlusions = trace_embree_occlusions(**embree, rays);
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_EQ((*cuda_occlusions)[index], embree_occlusions[index]);
        EXPECT_EQ((*cuda_occlusions)[index], expected[index]);
    }
}

TEST(CudaOcclusionParity, MatchesEmbreeForTheReferenceObjAndPlyMeshCorpus) {
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

    auto obj_cuda_occlusions = trace_cuda_occlusions(obj_device.scene, obj_device.bvh, rays);
    auto ply_cuda_occlusions = trace_cuda_occlusions(ply_device.scene, ply_device.bvh, rays);
    ASSERT_TRUE(obj_cuda_occlusions.has_value()) << obj_cuda_occlusions.error().message;
    ASSERT_TRUE(ply_cuda_occlusions.has_value()) << ply_cuda_occlusions.error().message;
    ASSERT_EQ(obj_cuda_occlusions->size(), rays.size());
    ASSERT_EQ(ply_cuda_occlusions->size(), rays.size());
    const auto obj_embree_occlusions = trace_embree_occlusions(**obj_embree, rays);
    const auto ply_embree_occlusions = trace_embree_occlusions(**ply_embree, rays);

    auto obj_occlusion_count = std::size_t{};
    auto ply_occlusion_count = std::size_t{};
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_EQ((*obj_cuda_occlusions)[index], obj_embree_occlusions[index]);
        EXPECT_EQ((*ply_cuda_occlusions)[index], ply_embree_occlusions[index]);
        EXPECT_EQ((*obj_cuda_occlusions)[index], (*ply_cuda_occlusions)[index]);
        obj_occlusion_count += (*obj_cuda_occlusions)[index] ? 1U : 0U;
        ply_occlusion_count += (*ply_cuda_occlusions)[index] ? 1U : 0U;
    }
    EXPECT_EQ(obj_occlusion_count, 16U);
    EXPECT_EQ(ply_occlusion_count, 16U);
}

} // namespace
} // namespace blackframe::engine
