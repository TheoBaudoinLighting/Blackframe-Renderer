#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/XPU/Shared/SceneBvhAbi.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace array = xpu::shared::scene_bvh_array;
using core::StatusCode;
using xpu::shared::SceneBvhBlas;
using xpu::shared::SceneBvhHeader;
using xpu::shared::SceneBvhInstanceReference;
using xpu::shared::SceneBvhNode;
using xpu::shared::SceneBvhNodeKind;

struct Box final {
    renderer::Point3 minimum{
        .x = std::numeric_limits<float>::infinity(),
        .y = std::numeric_limits<float>::infinity(),
        .z = std::numeric_limits<float>::infinity(),
    };
    renderer::Point3 maximum{
        .x = -std::numeric_limits<float>::infinity(),
        .y = -std::numeric_limits<float>::infinity(),
        .z = -std::numeric_limits<float>::infinity(),
    };
};

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{0};
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

[[nodiscard]] renderer::Matrix4 affine_matrix(const std::array<float, 12>& rows) {
    auto matrix = renderer::identity_matrix<renderer::TransportScalar>();
    for (auto row = std::size_t{0}; row < 3U; ++row) {
        for (auto column = std::size_t{0}; column < 4U; ++column) {
            matrix(row, column) = rows[row * 4U + column];
        }
    }
    return matrix;
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_mesh_a() {
    const auto positions = std::vector{
        renderer::Point3{.x = -4.0F, .y = -2.0F, .z = -1.0F},
        renderer::Point3{.x = -3.0F, .y = -2.0F, .z = -1.0F},
        renderer::Point3{.x = -4.0F, .y = -1.0F, .z = -1.0F},
        renderer::Point3{.x = -1.0F, .y = 1.0F, .z = 0.0F},
        renderer::Point3{.x = 0.0F, .y = 1.0F, .z = 0.0F},
        renderer::Point3{.x = -1.0F, .y = 2.0F, .z = 0.0F},
        renderer::Point3{.x = 2.0F, .y = -3.0F, .z = 1.0F},
        renderer::Point3{.x = 3.0F, .y = -3.0F, .z = 1.0F},
        renderer::Point3{.x = 2.0F, .y = -2.0F, .z = 1.0F},
        renderer::Point3{.x = 5.0F, .y = 4.0F, .z = 2.0F},
        renderer::Point3{.x = 6.0F, .y = 4.0F, .z = 2.0F},
        renderer::Point3{.x = 5.0F, .y = 5.0F, .z = 2.0F},
        renderer::Point3{.x = 8.0F, .y = 0.0F, .z = 3.0F},
        renderer::Point3{.x = 9.0F, .y = 0.0F, .z = 3.0F},
        renderer::Point3{.x = 8.0F, .y = 1.0F, .z = 3.0F},
    };
    auto triangles = std::vector<TriangleVertexIndices>{};
    triangles.reserve(5U);
    for (auto triangle = std::uint32_t{0}; triangle < 5U; ++triangle) {
        triangles.push_back(TriangleVertexIndices{
            .vertices = {triangle * 3U, triangle * 3U + 1U, triangle * 3U + 2U},
        });
    }
    auto mesh = TriangleMesh::create(
        positions,
        std::vector(positions.size(), renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}),
        std::vector(positions.size(), renderer::Point2{}), std::move(triangles));
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_mesh_b() {
    const auto positions = std::vector{
        renderer::Point3{.x = -2.0F, .y = -1.0F, .z = -2.0F},
        renderer::Point3{.x = 1.0F, .y = -1.0F, .z = -2.0F},
        renderer::Point3{.x = -2.0F, .y = 2.0F, .z = -2.0F},
        renderer::Point3{.x = 2.0F, .y = 2.0F, .z = 1.0F},
        renderer::Point3{.x = 4.0F, .y = 2.0F, .z = 1.0F},
        renderer::Point3{.x = 2.0F, .y = 3.0F, .z = 1.0F},
    };
    auto mesh = TriangleMesh::create(
        positions,
        std::vector(positions.size(), renderer::Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}),
        std::vector(positions.size(), renderer::Point2{}),
        std::vector{
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {3U, 4U, 5U}},
        });
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] std::shared_ptr<const TriangleMesh>
make_single_triangle_mesh(const std::uint32_t duplicate_count = 1U) {
    const auto positions = std::vector{
        renderer::Point3{.x = 4097.0F, .y = 0.0F, .z = 0.0F},
        renderer::Point3{.x = 4097.0F, .y = 1.0F, .z = 0.0F},
        renderer::Point3{.x = 4097.0F, .y = 0.0F, .z = 1.0F},
    };
    auto triangles = std::vector<TriangleVertexIndices>(
        duplicate_count, TriangleVertexIndices{.vertices = {0U, 1U, 2U}});
    auto mesh = TriangleMesh::create(
        positions,
        std::vector(positions.size(), renderer::Normal3{.x = 1.0F, .y = 0.0F, .z = 0.0F}),
        std::vector(positions.size(), renderer::Point2{}), std::move(triangles));
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] FrameSceneHandle make_single_geometry_scene(std::shared_ptr<const TriangleMesh> mesh,
                                                          const renderer::Matrix4 transform) {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{.id = {.value = 2U}, .mesh = std::move(mesh)}},
        .materials = {SceneMaterial{.id = {.value = 3U}, .spectral = std::nullopt}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 4U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 2U},
                    .material = {.value = 3U},
                    .local_to_parent = transform,
                    .visibility_mask = renderer::AllRayVisibility,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] FrameSceneDescription make_description(const bool permuted) {
    auto description = FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 300U}},
                SceneObject{.id = {.value = 100U}},
                SceneObject{.id = {.value = 200U}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 7U}, .mesh = make_mesh_a()},
                SceneGeometry{.id = {.value = 3U}, .mesh = make_mesh_b()},
            },
        .materials = {SceneMaterial{.id = {.value = 11U}, .spectral = std::nullopt}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 20U},
                    .parent = renderer::InstanceId{.value = 10U},
                    .object = {.value = 200U},
                    .geometry = {.value = 7U},
                    .material = {.value = 11U},
                    .local_to_parent = affine_matrix(
                        {1.0F, 0.0F, 0.0F, -4.0F, 0.0F, 3.0F, 0.0F, 2.0F, 0.0F, 0.0F, 1.0F, -1.0F}),
                    .visibility_mask = 0x0000FFFFU,
                },
                SceneInstance{
                    .id = {.value = 30U},
                    .parent = std::nullopt,
                    .object = {.value = 300U},
                    .geometry = {.value = 7U},
                    .material = {.value = 11U},
                    .local_to_parent = affine_matrix(
                        {0.0F, -1.0F, 0.0F, 10.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 2.0F}),
                    .visibility_mask = renderer::AllRayVisibility,
                },
                SceneInstance{
                    .id = {.value = 10U},
                    .parent = renderer::InstanceId{.value = 30U},
                    .object = {.value = 100U},
                    .geometry = {.value = 3U},
                    .material = {.value = 11U},
                    .local_to_parent = affine_matrix(
                        {2.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, -2.0F, 0.0F, 0.0F, -1.0F, 3.0F}),
                    .visibility_mask = 0xFF00FF00U,
                },
            },
    };
    if (permuted) {
        std::ranges::reverse(description.objects);
        std::ranges::reverse(description.geometries);
        std::ranges::reverse(description.materials);
        std::ranges::reverse(description.instances);
    }
    return description;
}

[[nodiscard]] FrameSceneHandle make_scene(const bool permuted = false) {
    auto result = FrameScene::create(make_description(permuted));
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return *result;
}

[[nodiscard]] CudaSceneSoA upload(const FrameScene& scene) {
    auto result = CudaSceneSoA::upload(scene);
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] CudaSceneBvh build(const CudaSceneSoA& scene) {
    auto result = CudaSceneBvh::build(scene);
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] std::vector<std::uint8_t> download(const CudaSceneBvh& bvh) {
    auto result = std::vector<std::uint8_t>(bvh.size_bytes());
    const auto status =
        cudaMemcpy(result.data(), bvh.device_data(), result.size(), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error{cudaGetErrorString(status)};
    }
    return result;
}

[[nodiscard]] SceneBvhHeader read_header(const std::span<const std::uint8_t> bytes) {
    if (bytes.size() < sizeof(SceneBvhHeader)) {
        throw std::runtime_error{"Downloaded CUDA BVH is smaller than its ABI header."};
    }
    auto header = SceneBvhHeader{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
}

template <typename Value>
[[nodiscard]] std::vector<Value> read_array(const std::span<const std::uint8_t> bytes,
                                            const SceneBvhHeader& header,
                                            const std::uint32_t index) {
    static_assert(std::is_trivially_copyable_v<Value>);
    if (index >= array::count) {
        throw std::runtime_error{"CUDA BVH array index is outside the frozen schema."};
    }
    const auto& descriptor = header.arrays[index];
    if (descriptor.element_size != sizeof(Value) ||
        descriptor.element_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error{"CUDA BVH array type disagrees with its descriptor."};
    }
    auto result = std::vector<Value>(static_cast<std::size_t>(descriptor.element_count));
    if (result.empty()) {
        if (descriptor.offset_bytes != 0U) {
            throw std::runtime_error{"Empty CUDA BVH array has a non-zero offset."};
        }
        return result;
    }
    const auto byte_count = result.size() * sizeof(Value);
    if (descriptor.offset_bytes > bytes.size() ||
        byte_count > bytes.size() - static_cast<std::size_t>(descriptor.offset_bytes)) {
        throw std::runtime_error{"CUDA BVH array exceeds the downloaded blob."};
    }
    std::memcpy(result.data(), bytes.data() + descriptor.offset_bytes, byte_count);
    return result;
}

void extend(Box& box, const renderer::Point3 point) {
    box.minimum.x = std::min(box.minimum.x, point.x);
    box.minimum.y = std::min(box.minimum.y, point.y);
    box.minimum.z = std::min(box.minimum.z, point.z);
    box.maximum.x = std::max(box.maximum.x, point.x);
    box.maximum.y = std::max(box.maximum.y, point.y);
    box.maximum.z = std::max(box.maximum.z, point.z);
}

[[nodiscard]] Box node_box(const SceneBvhNode& node) {
    return Box{
        .minimum = {.x = node.minimum_x, .y = node.minimum_y, .z = node.minimum_z},
        .maximum = {.x = node.maximum_x, .y = node.maximum_y, .z = node.maximum_z},
    };
}

[[nodiscard]] Box mesh_box(const TriangleMesh& mesh) {
    auto result = Box{};
    for (const auto point : mesh.positions()) {
        extend(result, point);
    }
    return result;
}

[[nodiscard]] Box triangle_box(const TriangleMesh& mesh, const std::uint32_t triangle_index) {
    auto result = Box{};
    const auto& triangle = mesh.triangles()[triangle_index];
    for (const auto vertex : triangle.vertices) {
        extend(result, mesh.positions()[vertex]);
    }
    return result;
}

void expect_exact_box(const Box& actual, const Box& expected) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.minimum.x),
              std::bit_cast<std::uint32_t>(expected.minimum.x));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.minimum.y),
              std::bit_cast<std::uint32_t>(expected.minimum.y));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.minimum.z),
              std::bit_cast<std::uint32_t>(expected.minimum.z));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.maximum.x),
              std::bit_cast<std::uint32_t>(expected.maximum.x));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.maximum.y),
              std::bit_cast<std::uint32_t>(expected.maximum.y));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.maximum.z),
              std::bit_cast<std::uint32_t>(expected.maximum.z));
}

[[nodiscard]] bool contains(const Box& box, const renderer::Point3 point) {
    return point.x >= box.minimum.x && point.y >= box.minimum.y && point.z >= box.minimum.z &&
           point.x <= box.maximum.x && point.y <= box.maximum.y && point.z <= box.maximum.z;
}

[[nodiscard]] double exact_transform_component(const renderer::Matrix4& matrix,
                                               const std::size_t row,
                                               const renderer::Point3 point) {
    const auto source = std::array{point.x, point.y, point.z};
    auto result = static_cast<double>(matrix(row, 3));
    for (auto column = std::size_t{0}; column < 3U; ++column) {
        result += static_cast<double>(matrix(row, column)) * static_cast<double>(source[column]);
    }
    return result;
}

[[nodiscard]] std::uint64_t normalized_hash(const std::span<const std::uint8_t> bytes) {
    auto hash = xpu::shared::SceneBvhFnv1aOffsetBasis;
    constexpr auto hash_begin = xpu::shared::SceneBvhContentHashOffset;
    constexpr auto hash_end = hash_begin + sizeof(std::uint64_t);
    for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
        const auto value = index >= hash_begin && index < hash_end ? std::uint8_t{0} : bytes[index];
        hash ^= value;
        hash *= xpu::shared::SceneBvhFnv1aPrime;
    }
    return hash;
}

struct ParsedBvh final {
    std::vector<std::uint8_t> bytes;
    SceneBvhHeader header;
    std::vector<SceneBvhBlas> blases;
    std::vector<SceneBvhNode> blas_nodes;
    std::vector<std::uint32_t> primitive_references;
    std::vector<SceneBvhNode> tlas_nodes;
    std::vector<SceneBvhInstanceReference> instance_references;
};

[[nodiscard]] ParsedBvh parse(const CudaSceneBvh& bvh) {
    auto bytes = download(bvh);
    const auto header = read_header(bytes);
    auto blases = read_array<SceneBvhBlas>(bytes, header, array::blas);
    auto blas_nodes = read_array<SceneBvhNode>(bytes, header, array::blas_node);
    auto primitive_references =
        read_array<std::uint32_t>(bytes, header, array::primitive_reference);
    auto tlas_nodes = read_array<SceneBvhNode>(bytes, header, array::tlas_node);
    auto instance_references =
        read_array<SceneBvhInstanceReference>(bytes, header, array::instance_reference);
    return ParsedBvh{
        .bytes = std::move(bytes),
        .header = header,
        .blases = std::move(blases),
        .blas_nodes = std::move(blas_nodes),
        .primitive_references = std::move(primitive_references),
        .tlas_nodes = std::move(tlas_nodes),
        .instance_references = std::move(instance_references),
    };
}

void validate_tree(const std::span<const SceneBvhNode> nodes, const std::uint32_t node_begin,
                   const std::uint32_t node_count, const std::uint32_t root,
                   const std::uint32_t reference_begin, const std::uint32_t reference_count) {
    ASSERT_GT(node_count, 0U);
    ASSERT_EQ(node_count, reference_count * 2U - 1U);
    ASSERT_GE(root, node_begin);
    ASSERT_LT(root, node_begin + node_count);

    auto state = std::vector<std::uint8_t>(nodes.size(), 0U);
    auto references_seen = std::vector<std::uint32_t>(reference_count, 0U);
    std::function<void(std::uint32_t)> visit = [&](const std::uint32_t index) {
        if (index < node_begin || index >= node_begin + node_count) {
            ADD_FAILURE() << "BVH child " << index << " escapes its tree's node range.";
            return;
        }
        if (state[index] == 1U) {
            ADD_FAILURE() << "BVH cycle reaches node " << index << '.';
            return;
        }
        if (state[index] == 2U) {
            ADD_FAILURE() << "BVH node " << index << " has multiple parents.";
            return;
        }
        state[index] = 1U;
        const auto& node = nodes[index];
        EXPECT_TRUE(std::isfinite(node.minimum_x));
        EXPECT_TRUE(std::isfinite(node.minimum_y));
        EXPECT_TRUE(std::isfinite(node.minimum_z));
        EXPECT_TRUE(std::isfinite(node.maximum_x));
        EXPECT_TRUE(std::isfinite(node.maximum_y));
        EXPECT_TRUE(std::isfinite(node.maximum_z));
        EXPECT_LE(node.minimum_x, node.maximum_x);
        EXPECT_LE(node.minimum_y, node.maximum_y);
        EXPECT_LE(node.minimum_z, node.maximum_z);

        if (node.kind == static_cast<std::uint32_t>(SceneBvhNodeKind::leaf)) {
            EXPECT_EQ(node.split_axis, xpu::shared::SceneBvhInvalidIndex);
            EXPECT_EQ(node.first_child, xpu::shared::SceneBvhInvalidIndex);
            EXPECT_EQ(node.second_child, xpu::shared::SceneBvhInvalidIndex);
            EXPECT_EQ(node.reference_count, 1U);
            if (node.reference_offset < reference_begin ||
                node.reference_offset >= reference_begin + reference_count) {
                ADD_FAILURE() << "BVH leaf reference escapes its tree's reference range.";
            } else {
                ++references_seen[node.reference_offset - reference_begin];
            }
        } else if (node.kind == static_cast<std::uint32_t>(SceneBvhNodeKind::internal)) {
            EXPECT_LT(node.split_axis, 3U);
            EXPECT_EQ(node.reference_count, 0U);
            EXPECT_NE(node.first_child, node.second_child);
            visit(node.first_child);
            visit(node.second_child);
            if (node.first_child >= node_begin && node.first_child < node_begin + node_count &&
                node.second_child >= node_begin && node.second_child < node_begin + node_count) {
                const auto first = node_box(nodes[node.first_child]);
                const auto second = node_box(nodes[node.second_child]);
                const auto expected = Box{
                    .minimum =
                        {
                            .x = std::min(first.minimum.x, second.minimum.x),
                            .y = std::min(first.minimum.y, second.minimum.y),
                            .z = std::min(first.minimum.z, second.minimum.z),
                        },
                    .maximum =
                        {
                            .x = std::max(first.maximum.x, second.maximum.x),
                            .y = std::max(first.maximum.y, second.maximum.y),
                            .z = std::max(first.maximum.z, second.maximum.z),
                        },
                };
                expect_exact_box(node_box(node), expected);
            }
        } else {
            ADD_FAILURE() << "BVH node has an unknown kind " << node.kind << '.';
        }
        state[index] = 2U;
    };
    visit(root);

    for (auto index = node_begin; index < node_begin + node_count; ++index) {
        EXPECT_EQ(state[index], 2U) << "Unreachable BVH node " << index << '.';
    }
    for (auto reference = std::uint32_t{0}; reference < reference_count; ++reference) {
        EXPECT_EQ(references_seen[reference], 1U)
            << "BVH reference " << reference_begin + reference << " is not used exactly once.";
    }
}

TEST(CudaSceneBvh, BlasBoundsExactlyCoverSourceTriangles) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_scene();
    auto uploaded = upload(*scene);
    auto bvh = build(uploaded);
    const auto parsed = parse(bvh);

    ASSERT_EQ(xpu::shared::validate_scene_bvh_header(parsed.header),
              xpu::shared::SceneBvhHeaderValidationStatus::valid);
    ASSERT_EQ(parsed.blases.size(), scene->geometries().size());
    for (const auto& blas : parsed.blases) {
        const auto geometry_result =
            scene->geometry(renderer::GeometryId{.value = blas.geometry_id});
        ASSERT_TRUE(geometry_result) << geometry_result.error().message;
        const auto& mesh = *geometry_result->get().mesh;
        ASSERT_LT(blas.root_node, parsed.blas_nodes.size());
        expect_exact_box(node_box(parsed.blas_nodes[blas.root_node]), mesh_box(mesh));

        auto leaf_count = std::uint32_t{0};
        for (auto index = blas.node_offset; index < blas.node_offset + blas.node_count; ++index) {
            const auto& node = parsed.blas_nodes[index];
            if (node.kind != static_cast<std::uint32_t>(SceneBvhNodeKind::leaf)) {
                continue;
            }
            ++leaf_count;
            ASSERT_EQ(node.reference_count, 1U);
            ASSERT_LT(node.reference_offset, parsed.primitive_references.size());
            const auto triangle = parsed.primitive_references[node.reference_offset];
            ASSERT_LT(triangle, mesh.triangles().size());
            expect_exact_box(node_box(node), triangle_box(mesh, triangle));
        }
        EXPECT_EQ(leaf_count, mesh.triangles().size());
    }
}

TEST(CudaSceneBvh, TlasBoundsConservativelyCoverHierarchicalInstances) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_scene();
    auto uploaded = upload(*scene);
    auto bvh = build(uploaded);
    const auto parsed = parse(bvh);

    ASSERT_EQ(scene->instances().size(), 3U);
    ASSERT_EQ(parsed.instance_references.size(), scene->instances().size());
    ASSERT_LT(parsed.header.tlas_root_node, parsed.tlas_nodes.size());
    const auto root_box = node_box(parsed.tlas_nodes[parsed.header.tlas_root_node]);
    for (const auto& node : parsed.tlas_nodes) {
        if (node.kind != static_cast<std::uint32_t>(SceneBvhNodeKind::leaf)) {
            continue;
        }
        ASSERT_EQ(node.reference_count, 1U);
        ASSERT_LT(node.reference_offset, parsed.instance_references.size());
        const auto& reference = parsed.instance_references[node.reference_offset];
        const auto instance_result =
            scene->instance(renderer::InstanceId{.value = reference.instance_id});
        ASSERT_TRUE(instance_result) << instance_result.error().message;
        const auto geometry_result =
            scene->geometry(renderer::GeometryId{.value = reference.geometry_id});
        ASSERT_TRUE(geometry_result) << geometry_result.error().message;
        const auto transform_result =
            scene->world_transform(renderer::InstanceId{.value = reference.instance_id});
        ASSERT_TRUE(transform_result) << transform_result.error().message;
        const auto& matrix = transform_result->get().matrix();
        const auto leaf_box = node_box(node);
        for (const auto point : geometry_result->get().mesh->positions()) {
            const auto world_point = transform_result->get().apply(point);
            EXPECT_TRUE(contains(leaf_box, world_point))
                << "TLAS leaf for instance " << reference.instance_id
                << " excludes a transformed mesh vertex.";
            EXPECT_TRUE(contains(root_box, world_point))
                << "TLAS root excludes a transformed mesh vertex.";
            const auto leaf_minimum =
                std::array{leaf_box.minimum.x, leaf_box.minimum.y, leaf_box.minimum.z};
            const auto leaf_maximum =
                std::array{leaf_box.maximum.x, leaf_box.maximum.y, leaf_box.maximum.z};
            for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
                const auto exact = exact_transform_component(matrix, axis, point);
                EXPECT_LE(static_cast<double>(leaf_minimum[axis]), exact);
                EXPECT_GE(static_cast<double>(leaf_maximum[axis]), exact);
            }
        }
    }
}

TEST(CudaSceneBvh, TlasBoundsCoverFusedAndUnfusedCancellation) {
    ASSERT_TRUE(select_test_device());
    const auto transform = affine_matrix(
        {4097.0F, 0.0F, 0.0F, -16785408.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F});
    const auto scene = make_single_geometry_scene(make_single_triangle_mesh(), transform);
    auto uploaded = upload(*scene);
    auto bvh = build(uploaded);
    const auto parsed = parse(bvh);

    ASSERT_EQ(parsed.tlas_nodes.size(), 1U);
    const auto bounds = node_box(parsed.tlas_nodes.front());
    EXPECT_LE(static_cast<double>(bounds.minimum.x), 0.0);
    EXPECT_GE(static_cast<double>(bounds.maximum.x), 1.0);
    for (const auto point : scene->geometries().front().mesh->positions()) {
        const auto exact = exact_transform_component(transform, 0U, point);
        ASSERT_EQ(exact, 1.0);
        EXPECT_LE(static_cast<double>(bounds.minimum.x), exact);
        EXPECT_GE(static_cast<double>(bounds.maximum.x), exact);
    }
}

TEST(CudaSceneBvh, EqualCentroidsUsePrimitiveIdentifiersAsTieBreaks) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_single_geometry_scene(
        make_single_triangle_mesh(3U), renderer::identity_matrix<renderer::TransportScalar>());
    auto uploaded = upload(*scene);
    auto bvh = build(uploaded);
    const auto parsed = parse(bvh);

    ASSERT_EQ(parsed.blases.size(), 1U);
    EXPECT_EQ(parsed.blases.front().node_count, 5U);
    EXPECT_EQ(parsed.primitive_references, (std::vector<std::uint32_t>{0U, 1U, 2U}));
}

TEST(CudaSceneBvh, BinaryTopologyIsAcyclicAndReferencesEveryLeafExactlyOnce) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_scene();
    auto uploaded = upload(*scene);
    auto bvh = build(uploaded);
    const auto parsed = parse(bvh);

    auto next_node = std::uint32_t{0};
    auto next_reference = std::uint32_t{0};
    for (auto blas_index = std::uint32_t{0}; blas_index < parsed.blases.size(); ++blas_index) {
        const auto& blas = parsed.blases[blas_index];
        EXPECT_EQ(blas.node_offset, next_node);
        EXPECT_EQ(blas.primitive_reference_offset, next_reference);
        validate_tree(parsed.blas_nodes, blas.node_offset, blas.node_count, blas.root_node,
                      blas.primitive_reference_offset, blas.primitive_reference_count);
        const auto geometry_result =
            scene->geometry(renderer::GeometryId{.value = blas.geometry_id});
        ASSERT_TRUE(geometry_result) << geometry_result.error().message;
        EXPECT_EQ(blas.primitive_reference_count, geometry_result->get().mesh->triangles().size());
        next_node += blas.node_count;
        next_reference += blas.primitive_reference_count;
    }
    EXPECT_EQ(next_node, parsed.blas_nodes.size());
    EXPECT_EQ(next_reference, parsed.primitive_references.size());
    validate_tree(parsed.tlas_nodes, 0U, static_cast<std::uint32_t>(parsed.tlas_nodes.size()),
                  parsed.header.tlas_root_node, 0U,
                  static_cast<std::uint32_t>(parsed.instance_references.size()));

    auto expected_instance_index = std::map<std::uint32_t, std::uint32_t>{};
    for (auto index = std::uint32_t{0}; index < scene->instances().size(); ++index) {
        expected_instance_index.emplace(scene->instances()[index].id.value, index);
    }
    auto expected_geometry_index = std::map<std::uint32_t, std::uint32_t>{};
    for (auto index = std::uint32_t{0}; index < scene->geometries().size(); ++index) {
        expected_geometry_index.emplace(scene->geometries()[index].id.value, index);
    }
    auto expected_blas_index = std::map<std::uint32_t, std::uint32_t>{};
    for (auto index = std::uint32_t{0}; index < parsed.blases.size(); ++index) {
        expected_blas_index.emplace(parsed.blases[index].geometry_id, index);
    }
    auto seen_instance_ids = std::map<std::uint32_t, std::uint32_t>{};
    for (const auto& reference : parsed.instance_references) {
        const auto instance_result =
            scene->instance(renderer::InstanceId{.value = reference.instance_id});
        ASSERT_TRUE(instance_result) << instance_result.error().message;
        const auto& instance = instance_result->get();
        ++seen_instance_ids[reference.instance_id];
        EXPECT_EQ(reference.scene_instance_index,
                  expected_instance_index.at(reference.instance_id));
        EXPECT_EQ(reference.geometry_id, instance.geometry.value);
        EXPECT_EQ(reference.scene_geometry_index,
                  expected_geometry_index.at(reference.geometry_id));
        EXPECT_EQ(reference.blas_index, expected_blas_index.at(reference.geometry_id));
        EXPECT_EQ(reference.visibility_mask, instance.visibility_mask);
        EXPECT_EQ(reference.reserved, (std::array<std::uint32_t, 2>{}));
    }
    ASSERT_EQ(seen_instance_ids.size(), scene->instances().size());
    for (const auto& [instance_id, count] : seen_instance_ids) {
        EXPECT_EQ(count, 1U) << "Instance " << instance_id << " is not referenced exactly once.";
    }
}

TEST(CudaSceneBvh, CanonicalBytesIgnoreInsertionAndAllocationIdentity) {
    ASSERT_TRUE(select_test_device());
    const auto first_scene = make_scene(false);
    const auto second_scene = make_scene(true);
    auto first_upload = upload(*first_scene);
    auto second_upload = upload(*second_scene);
    auto first = build(first_upload);
    auto second = build(second_upload);
    const auto first_bytes = download(first);
    const auto second_bytes = download(second);

    EXPECT_EQ(first.header().source_scene_hash, first_upload.header().content_hash);
    EXPECT_EQ(second.header().source_scene_hash, second_upload.header().content_hash);
    EXPECT_EQ(first.header().source_scene_hash, second.header().source_scene_hash);
    EXPECT_EQ(first.header().content_hash, second.header().content_hash);
    EXPECT_EQ(first.header().content_hash, normalized_hash(first_bytes));
    EXPECT_EQ(second.header().content_hash, normalized_hash(second_bytes));
    EXPECT_EQ(first_bytes, second_bytes);
}

TEST(CudaSceneBvh, EncodesEmptySceneAndRejectsBudgetOrAbiMismatch) {
    ASSERT_TRUE(select_test_device());
    const auto empty_scene_result = FrameScene::create(FrameSceneDescription{});
    ASSERT_TRUE(empty_scene_result) << empty_scene_result.error().message;
    auto empty_upload = upload(**empty_scene_result);
    auto empty_bvh = build(empty_upload);
    const auto parsed = parse(empty_bvh);
    EXPECT_EQ(parsed.bytes.size(), sizeof(SceneBvhHeader));
    EXPECT_EQ(parsed.header.blas_count, 0U);
    EXPECT_EQ(parsed.header.blas_node_count, 0U);
    EXPECT_EQ(parsed.header.primitive_reference_count, 0U);
    EXPECT_EQ(parsed.header.tlas_node_count, 0U);
    EXPECT_EQ(parsed.header.instance_reference_count, 0U);
    EXPECT_EQ(parsed.header.tlas_root_node, xpu::shared::SceneBvhInvalidIndex);
    EXPECT_EQ(xpu::shared::validate_scene_bvh_header(parsed.header),
              xpu::shared::SceneBvhHeaderValidationStatus::valid);
    for (const auto& descriptor : parsed.header.arrays) {
        EXPECT_EQ(descriptor.offset_bytes, 0U);
        EXPECT_EQ(descriptor.element_count, 0U);
        EXPECT_NE(descriptor.element_size, 0U);
    }
    auto malformed = parsed.header;
    malformed.instance_reference_count = 1U;
    EXPECT_EQ(xpu::shared::validate_scene_bvh_header(malformed),
              xpu::shared::SceneBvhHeaderValidationStatus::invalid_topology_counts);
    malformed = parsed.header;
    malformed.tlas_root_node = 0U;
    EXPECT_EQ(xpu::shared::validate_scene_bvh_header(malformed),
              xpu::shared::SceneBvhHeaderValidationStatus::invalid_tlas_root);

    const auto scene = make_scene();
    auto uploaded = upload(*scene);
    const auto rejected = CudaSceneBvh::build(
        uploaded,
        CudaSceneBvhBuildOptions{
            .device_memory_budget =
                xpu::cuda::DeviceMemoryBudget{.maximum_bytes = sizeof(SceneBvhHeader) - 1U},
        });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(rejected.error().message.find("explicit device-memory budget"), std::string::npos);

    const auto incompatible =
        CudaSceneBvh::build(uploaded, CudaSceneBvhBuildOptions{.abi_major = 2U});
    ASSERT_FALSE(incompatible);
    EXPECT_EQ(incompatible.error().code, StatusCode::incompatible);
}

} // namespace
} // namespace blackframe::engine
