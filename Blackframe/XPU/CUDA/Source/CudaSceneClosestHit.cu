#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/Shared/SceneBvhAbi.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The scene closest-hit kernel must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in CUDA scene traversal code."
#endif

namespace {

namespace shared = blackframe::xpu::shared;
namespace bvh_array = blackframe::xpu::shared::scene_bvh_array;
namespace scene_column = blackframe::xpu::shared::scene_soa_column;

using shared::ClosestHit;
using shared::SceneBvhArrayDescriptor;
using shared::SceneBvhBlas;
using shared::SceneBvhHeader;
using shared::SceneBvhInstanceReference;
using shared::SceneBvhNode;
using shared::SceneBvhNodeKind;
using shared::SceneClosestHitResult;
using shared::SceneClosestHitStatus;
using shared::SceneSoaColumnDescriptor;
using shared::SceneSoaHeader;
using shared::TransportRay;

constexpr auto ThreadsPerBlock = std::uint32_t{256U};
constexpr auto TraversalStackCapacity = std::uint32_t{64U};
constexpr auto MaximumU64 = ~std::uint64_t{0U};

struct Vector3 final {
    float x{};
    float y{};
    float z{};
};

struct TriangleIntersection final {
    bool hit{};
    float parameter{};
    float barycentric0{};
    float barycentric1{};
    float barycentric2{};
};

struct ClosestCandidate final {
    bool present{};
    float parameter{};
    std::uint32_t instance_id{shared::SceneBvhInvalidIndex};
    std::uint32_t primitive_id{shared::SceneBvhInvalidIndex};
    ClosestHit hit{};
};

[[nodiscard]] __device__ constexpr std::uint32_t
status_value(const SceneClosestHitStatus status) noexcept {
    return static_cast<std::uint32_t>(status);
}

[[nodiscard]] __device__ constexpr std::uint64_t align_up(const std::uint64_t value,
                                                          const std::uint64_t alignment) noexcept {
    const auto remainder = value % alignment;
    return remainder == 0U ? value : value + (alignment - remainder);
}

[[nodiscard]] __device__ const SceneSoaColumnDescriptor&
scene_descriptor(const SceneSoaHeader* const header, const std::uint32_t column) noexcept {
    const auto* const descriptors = reinterpret_cast<const SceneSoaColumnDescriptor*>(
        reinterpret_cast<const std::uint8_t*>(header) + offsetof(SceneSoaHeader, columns));
    return descriptors[column];
}

[[nodiscard]] __device__ const SceneBvhArrayDescriptor&
bvh_descriptor(const SceneBvhHeader* const header, const std::uint32_t array) noexcept {
    const auto* const descriptors = reinterpret_cast<const SceneBvhArrayDescriptor*>(
        reinterpret_cast<const std::uint8_t*>(header) + offsetof(SceneBvhHeader, arrays));
    return descriptors[array];
}

[[nodiscard]] __device__ std::uint64_t scene_column_count(const SceneSoaHeader& header,
                                                          const std::uint32_t column) noexcept {
    if (column == scene_column::object_id) {
        return header.object_count;
    }
    if (column >= scene_column::geometry_id && column <= scene_column::geometry_triangle_count) {
        return header.geometry_count;
    }
    if (column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) {
        return header.vertex_count;
    }
    if (column >= scene_column::triangle_vertex_0 && column <= scene_column::triangle_vertex_2) {
        return header.triangle_count;
    }
    if (column >= scene_column::material_id && column < scene_column::instance_id) {
        return header.material_count;
    }
    if (column >= scene_column::instance_id && column < scene_column::punctual_kind) {
        return header.instance_count;
    }
    if (column >= scene_column::punctual_kind &&
        column < scene_column::mesh_area_light_instance_id) {
        return header.punctual_light_count;
    }
    if (column == scene_column::mesh_area_light_instance_id) {
        return header.mesh_area_light_count;
    }
    if (column >= scene_column::environment_wavelength_nanometers && column < scene_column::count) {
        return header.environment_count;
    }
    return 0U;
}

[[nodiscard]] __device__ std::uint32_t
scene_column_element_size(const std::uint32_t column) noexcept {
    if (column >= scene_column::count) {
        return 0U;
    }
    if (column >= scene_column::geometry_vertex_offset &&
        column <= scene_column::geometry_triangle_count) {
        return sizeof(std::uint64_t);
    }
    if (column == scene_column::material_spectral_present ||
        (column >= scene_column::material_wavelength_measure &&
         column < scene_column::material_reflectance) ||
        column == scene_column::instance_parent_present ||
        (column >= scene_column::environment_wavelength_measure &&
         column < scene_column::environment_radiance)) {
        return sizeof(std::uint8_t);
    }
    if ((column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) ||
        (column >= scene_column::material_wavelength_nanometers &&
         column < scene_column::material_wavelength_measure) ||
        (column >= scene_column::material_reflectance && column < scene_column::instance_id) ||
        (column >= scene_column::instance_local_to_parent &&
         column < scene_column::punctual_kind) ||
        (column >= scene_column::punctual_position_x &&
         column < scene_column::mesh_area_light_instance_id) ||
        (column >= scene_column::environment_wavelength_nanometers &&
         column < scene_column::environment_wavelength_measure) ||
        (column >= scene_column::environment_radiance && column < scene_column::count)) {
        return sizeof(float);
    }
    return sizeof(std::uint32_t);
}

[[nodiscard]] __device__ SceneClosestHitStatus
validate_scene_layout(const std::uint8_t* const bytes, const std::size_t byte_count) noexcept {
    if (byte_count < sizeof(SceneSoaHeader)) {
        return SceneClosestHitStatus::invalid_scene;
    }
    const auto* const header = reinterpret_cast<const SceneSoaHeader*>(bytes);
    if (header->magic != shared::SceneSoaMagic || header->abi_major != shared::SceneSoaAbiMajor ||
        header->abi_minor != shared::SceneSoaAbiMinor ||
        header->header_size != sizeof(SceneSoaHeader) ||
        header->column_count != scene_column::count ||
        header->hash_algorithm != shared::SceneSoaHashAlgorithmFnv1a64 ||
        header->environment_count > 1U || header->total_size_bytes != byte_count ||
        header->total_size_bytes < sizeof(SceneSoaHeader)) {
        return SceneClosestHitStatus::invalid_scene;
    }

    const auto* const reserved =
        reinterpret_cast<const std::uint64_t*>(bytes + offsetof(SceneSoaHeader, reserved));
    for (auto index = std::uint32_t{0U}; index < 5U; ++index) {
        if (reserved[index] != 0U) {
            return SceneClosestHitStatus::invalid_scene;
        }
    }

    auto cursor = align_up(sizeof(SceneSoaHeader), shared::SceneSoaColumnAlignment);
    for (auto column = std::uint32_t{0U}; column < scene_column::count; ++column) {
        const auto& descriptor = scene_descriptor(header, column);
        const auto expected_count = scene_column_count(*header, column);
        const auto expected_size = scene_column_element_size(column);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return SceneClosestHitStatus::invalid_scene;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return SceneClosestHitStatus::invalid_scene;
            }
            continue;
        }
        if (cursor > MaximumU64 - (shared::SceneSoaColumnAlignment - 1U)) {
            return SceneClosestHitStatus::invalid_scene;
        }
        cursor = align_up(cursor, shared::SceneSoaColumnAlignment);
        if (descriptor.offset_bytes != cursor || expected_count > MaximumU64 / expected_size) {
            return SceneClosestHitStatus::invalid_scene;
        }
        const auto bytes_in_column = expected_count * expected_size;
        if (cursor > MaximumU64 - bytes_in_column) {
            return SceneClosestHitStatus::invalid_scene;
        }
        cursor += bytes_in_column;
    }
    return cursor == header->total_size_bytes ? SceneClosestHitStatus::miss
                                              : SceneClosestHitStatus::invalid_scene;
}

[[nodiscard]] __device__ std::uint64_t bvh_array_count(const SceneBvhHeader& header,
                                                       const std::uint32_t array) noexcept {
    switch (array) {
    case bvh_array::blas:
        return header.blas_count;
    case bvh_array::blas_node:
        return header.blas_node_count;
    case bvh_array::primitive_reference:
        return header.primitive_reference_count;
    case bvh_array::tlas_node:
        return header.tlas_node_count;
    case bvh_array::instance_reference:
        return header.instance_reference_count;
    default:
        return 0U;
    }
}

[[nodiscard]] __device__ std::uint32_t bvh_array_element_size(const std::uint32_t array) noexcept {
    switch (array) {
    case bvh_array::blas:
        return sizeof(SceneBvhBlas);
    case bvh_array::blas_node:
    case bvh_array::tlas_node:
        return sizeof(SceneBvhNode);
    case bvh_array::primitive_reference:
        return sizeof(std::uint32_t);
    case bvh_array::instance_reference:
        return sizeof(SceneBvhInstanceReference);
    default:
        return 0U;
    }
}

[[nodiscard]] __device__ SceneClosestHitStatus
validate_bvh_layout(const std::uint8_t* const bytes, const std::size_t byte_count,
                    const SceneSoaHeader& scene) noexcept {
    if (byte_count < sizeof(SceneBvhHeader)) {
        return SceneClosestHitStatus::invalid_bvh;
    }
    const auto* const header = reinterpret_cast<const SceneBvhHeader*>(bytes);
    if (header->magic != shared::SceneBvhMagic || header->abi_major != shared::SceneBvhAbiMajor ||
        header->abi_minor != shared::SceneBvhAbiMinor ||
        header->header_size != sizeof(SceneBvhHeader) || header->array_count != bvh_array::count ||
        header->hash_algorithm != shared::SceneBvhHashAlgorithmFnv1a64 ||
        header->leaf_capacity != shared::SceneBvhLeafCapacity ||
        header->total_size_bytes != byte_count ||
        header->total_size_bytes < sizeof(SceneBvhHeader) ||
        header->source_scene_hash != scene.content_hash ||
        header->blas_count != scene.geometry_count ||
        header->primitive_reference_count != scene.triangle_count ||
        header->instance_reference_count != scene.instance_count ||
        header->blas_count > shared::SceneBvhInvalidIndex ||
        header->blas_node_count > shared::SceneBvhInvalidIndex ||
        header->primitive_reference_count > shared::SceneBvhInvalidIndex ||
        header->tlas_node_count > shared::SceneBvhInvalidIndex ||
        header->instance_reference_count > shared::SceneBvhInvalidIndex) {
        return SceneClosestHitStatus::invalid_bvh;
    }

    if ((header->blas_count == 0U &&
         (header->blas_node_count != 0U || header->primitive_reference_count != 0U)) ||
        (header->blas_count != 0U &&
         (header->primitive_reference_count < header->blas_count ||
          header->blas_node_count !=
              header->primitive_reference_count * 2U - header->blas_count)) ||
        (header->instance_reference_count == 0U && header->tlas_node_count != 0U) ||
        (header->instance_reference_count != 0U &&
         header->tlas_node_count != header->instance_reference_count * 2U - 1U) ||
        (header->tlas_node_count == 0U && header->tlas_root_node != shared::SceneBvhInvalidIndex) ||
        (header->tlas_node_count != 0U && header->tlas_root_node != 0U)) {
        return SceneClosestHitStatus::invalid_bvh;
    }

    const auto* const reserved =
        reinterpret_cast<const std::uint64_t*>(bytes + offsetof(SceneBvhHeader, reserved));
    for (auto index = std::uint32_t{0U}; index < 5U; ++index) {
        if (reserved[index] != 0U) {
            return SceneClosestHitStatus::invalid_bvh;
        }
    }

    auto cursor = align_up(sizeof(SceneBvhHeader), shared::SceneBvhArrayAlignment);
    for (auto array = std::uint32_t{0U}; array < bvh_array::count; ++array) {
        const auto& descriptor = bvh_descriptor(header, array);
        const auto expected_count = bvh_array_count(*header, array);
        const auto expected_size = bvh_array_element_size(array);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return SceneClosestHitStatus::invalid_bvh;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return SceneClosestHitStatus::invalid_bvh;
            }
            continue;
        }
        if (cursor > MaximumU64 - (shared::SceneBvhArrayAlignment - 1U)) {
            return SceneClosestHitStatus::invalid_bvh;
        }
        cursor = align_up(cursor, shared::SceneBvhArrayAlignment);
        if (descriptor.offset_bytes != cursor || expected_count > MaximumU64 / expected_size) {
            return SceneClosestHitStatus::invalid_bvh;
        }
        const auto bytes_in_array = expected_count * expected_size;
        if (cursor > MaximumU64 - bytes_in_array) {
            return SceneClosestHitStatus::invalid_bvh;
        }
        cursor += bytes_in_array;
    }
    return cursor == header->total_size_bytes ? SceneClosestHitStatus::miss
                                              : SceneClosestHitStatus::invalid_bvh;
}

template <typename Value>
[[nodiscard]] __device__ const Value* scene_values(const std::uint8_t* const bytes,
                                                   const SceneSoaHeader& header,
                                                   const std::uint32_t column) noexcept {
    return reinterpret_cast<const Value*>(bytes + scene_descriptor(&header, column).offset_bytes);
}

template <typename Value>
[[nodiscard]] __device__ const Value* bvh_values(const std::uint8_t* const bytes,
                                                 const SceneBvhHeader& header,
                                                 const std::uint32_t array) noexcept {
    return reinterpret_cast<const Value*>(bytes + bvh_descriptor(&header, array).offset_bytes);
}

[[nodiscard]] __device__ bool finite_vector(const Vector3 value) noexcept {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

[[nodiscard]] __device__ float component(const Vector3 value, const std::uint32_t axis) noexcept {
    return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
}

[[nodiscard]] __device__ Vector3 subtract(const Vector3 left, const Vector3 right) noexcept {
    return Vector3{.x = left.x - right.x, .y = left.y - right.y, .z = left.z - right.z};
}

[[nodiscard]] __device__ bool valid_ray(const TransportRay& ray) noexcept {
    const auto origin = Vector3{.x = ray.origin_x, .y = ray.origin_y, .z = ray.origin_z};
    const auto direction =
        Vector3{.x = ray.direction_x, .y = ray.direction_y, .z = ray.direction_z};
    return finite_vector(origin) && finite_vector(direction) &&
           (direction.x != 0.0F || direction.y != 0.0F || direction.z != 0.0F) &&
           isfinite(ray.t_min) && ray.t_min >= 0.0F && !isnan(ray.t_max) &&
           ray.t_max >= ray.t_min && isfinite(ray.time) && ray.time >= 0.0F && ray.time <= 1.0F &&
           ray.reserved == 0U;
}

[[nodiscard]] __device__ SceneClosestHitStatus intersect_bounds(
    const SceneBvhNode& node, const Vector3 origin, const Vector3 direction, const float t_min,
    const float t_max, bool& intersects, float& near_parameter) noexcept {
    const auto minimum = Vector3{.x = node.minimum_x, .y = node.minimum_y, .z = node.minimum_z};
    const auto maximum = Vector3{.x = node.maximum_x, .y = node.maximum_y, .z = node.maximum_z};
    if (!finite_vector(minimum) || !finite_vector(maximum) || minimum.x > maximum.x ||
        minimum.y > maximum.y || minimum.z > maximum.z) {
        return SceneClosestHitStatus::invalid_topology;
    }

    // Evaluate slabs in double and expand every quotient by one representable
    // step. Serialized BVH bounds are exact float enclosures, so a rounded
    // float division must not reject a triangle that lies on a box boundary.
    auto near_value = static_cast<double>(t_min);
    auto far_value = static_cast<double>(t_max);
    for (auto axis = std::uint32_t{0U}; axis < 3U; ++axis) {
        const auto axis_origin = component(origin, axis);
        const auto axis_direction = component(direction, axis);
        const auto axis_minimum = component(minimum, axis);
        const auto axis_maximum = component(maximum, axis);
        if (axis_direction == 0.0F) {
            if (axis_origin < axis_minimum || axis_origin > axis_maximum) {
                intersects = false;
                return SceneClosestHitStatus::miss;
            }
            continue;
        }

        const auto first = (static_cast<double>(axis_minimum) - static_cast<double>(axis_origin)) /
                           static_cast<double>(axis_direction);
        const auto second = (static_cast<double>(axis_maximum) - static_cast<double>(axis_origin)) /
                            static_cast<double>(axis_direction);
        if (isnan(first) || isnan(second)) {
            return SceneClosestHitStatus::numerical_failure;
        }
        const auto infinity = static_cast<double>(INFINITY);
        const auto axis_near = nextafter(fmin(first, second), -infinity);
        const auto axis_far = nextafter(fmax(first, second), infinity);
        near_value = fmax(near_value, axis_near);
        far_value = fmin(far_value, axis_far);
        if (near_value > far_value) {
            intersects = false;
            return SceneClosestHitStatus::miss;
        }
    }
    intersects = true;
    near_parameter = static_cast<float>(near_value);
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ bool push_node(std::uint32_t* const stack, std::uint32_t& size,
                                        const std::uint32_t node) noexcept {
    if (size >= TraversalStackCapacity) {
        return false;
    }
    stack[size++] = node;
    return true;
}

[[nodiscard]] __device__ SceneClosestHitStatus
push_intersecting_children(const SceneBvhNode* const nodes, const std::uint32_t first_child,
                           const std::uint32_t second_child, const std::uint32_t node_begin,
                           const std::uint32_t node_count, const Vector3 origin,
                           const Vector3 direction, const float t_min, const float t_max,
                           std::uint32_t* const stack, std::uint32_t& stack_size) noexcept {
    const auto node_end = static_cast<std::uint64_t>(node_begin) + node_count;
    if (first_child == second_child || first_child < node_begin || second_child < node_begin ||
        first_child >= node_end || second_child >= node_end) {
        return SceneClosestHitStatus::invalid_topology;
    }

    auto first_hit = false;
    auto second_hit = false;
    auto first_near = 0.0F;
    auto second_near = 0.0F;
    auto status = intersect_bounds(nodes[first_child], origin, direction, t_min, t_max, first_hit,
                                   first_near);
    if (status != SceneClosestHitStatus::miss) {
        return status;
    }
    status = intersect_bounds(nodes[second_child], origin, direction, t_min, t_max, second_hit,
                              second_near);
    if (status != SceneClosestHitStatus::miss) {
        return status;
    }
    if (!first_hit && !second_hit) {
        return SceneClosestHitStatus::miss;
    }
    if (first_hit && !second_hit) {
        return push_node(stack, stack_size, first_child) ? SceneClosestHitStatus::miss
                                                         : SceneClosestHitStatus::stack_overflow;
    }
    if (!first_hit && second_hit) {
        return push_node(stack, stack_size, second_child) ? SceneClosestHitStatus::miss
                                                          : SceneClosestHitStatus::stack_overflow;
    }

    const auto first_is_near =
        first_near < second_near || (first_near == second_near && first_child < second_child);
    const auto near_child = first_is_near ? first_child : second_child;
    const auto far_child = first_is_near ? second_child : first_child;
    if (!push_node(stack, stack_size, far_child) || !push_node(stack, stack_size, near_child)) {
        return SceneClosestHitStatus::stack_overflow;
    }
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ float matrix_value(const std::uint8_t* const scene_bytes,
                                            const SceneSoaHeader& scene,
                                            const std::uint32_t first_column,
                                            const std::uint32_t instance,
                                            const std::uint32_t element) noexcept {
    return scene_values<float>(scene_bytes, scene, first_column + element)[instance];
}

[[nodiscard]] __device__ SceneClosestHitStatus load_matrix(const std::uint8_t* const scene_bytes,
                                                           const SceneSoaHeader& scene,
                                                           const std::uint32_t first_column,
                                                           const std::uint32_t instance,
                                                           float* const matrix) noexcept {
    for (auto element = std::uint32_t{0U}; element < shared::SceneSoaMatrixElementCount;
         ++element) {
        matrix[element] = matrix_value(scene_bytes, scene, first_column, instance, element);
        if (!isfinite(matrix[element])) {
            return SceneClosestHitStatus::numerical_failure;
        }
    }
    if (matrix[12] != 0.0F || matrix[13] != 0.0F || matrix[14] != 0.0F || matrix[15] != 1.0F) {
        return SceneClosestHitStatus::invalid_topology;
    }
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ Vector3 transform_point(const float* const matrix,
                                                 const Vector3 point) noexcept {
    return Vector3{
        .x =
            fmaf(matrix[0], point.x, fmaf(matrix[1], point.y, fmaf(matrix[2], point.z, matrix[3]))),
        .y =
            fmaf(matrix[4], point.x, fmaf(matrix[5], point.y, fmaf(matrix[6], point.z, matrix[7]))),
        .z = fmaf(matrix[8], point.x,
                  fmaf(matrix[9], point.y, fmaf(matrix[10], point.z, matrix[11]))),
    };
}

[[nodiscard]] __device__ Vector3 transform_vector(const float* const matrix,
                                                  const Vector3 vector) noexcept {
    return Vector3{
        .x = fmaf(matrix[0], vector.x, fmaf(matrix[1], vector.y, matrix[2] * vector.z)),
        .y = fmaf(matrix[4], vector.x, fmaf(matrix[5], vector.y, matrix[6] * vector.z)),
        .z = fmaf(matrix[8], vector.x, fmaf(matrix[9], vector.y, matrix[10] * vector.z)),
    };
}

[[nodiscard]] __device__ SceneClosestHitStatus
object_space_ray(const std::uint8_t* const scene_bytes, const SceneSoaHeader& scene,
                 const std::uint32_t instance, const TransportRay& world_ray, Vector3& origin,
                 Vector3& direction) noexcept {
    float matrix[shared::SceneSoaMatrixElementCount]{};
    auto status =
        load_matrix(scene_bytes, scene, scene_column::instance_world_to_local, instance, matrix);
    if (status != SceneClosestHitStatus::miss) {
        return status;
    }
    origin = transform_point(
        matrix, Vector3{.x = world_ray.origin_x, .y = world_ray.origin_y, .z = world_ray.origin_z});
    direction = transform_vector(matrix, Vector3{.x = world_ray.direction_x,
                                                 .y = world_ray.direction_y,
                                                 .z = world_ray.direction_z});
    if (!finite_vector(origin) || !finite_vector(direction) ||
        (direction.x == 0.0F && direction.y == 0.0F && direction.z == 0.0F)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ SceneClosestHitStatus refine_ambiguous_triangle_edges(
    const Vector3 vertex0, const Vector3 vertex1, const Vector3 vertex2, const Vector3 origin,
    const Vector3 direction, const std::uint32_t kx, const std::uint32_t ky, const std::uint32_t kz,
    const float t_min, const float t_max, TriangleIntersection& result) noexcept {
    const auto direction_x = static_cast<double>(component(direction, kx));
    const auto direction_y = static_cast<double>(component(direction, ky));
    const auto direction_z = static_cast<double>(component(direction, kz));
    if (direction_z == 0.0) {
        return SceneClosestHitStatus::numerical_failure;
    }

    const auto point0_x =
        static_cast<double>(component(vertex0, kx)) - static_cast<double>(component(origin, kx));
    const auto point0_y =
        static_cast<double>(component(vertex0, ky)) - static_cast<double>(component(origin, ky));
    const auto point0_z =
        static_cast<double>(component(vertex0, kz)) - static_cast<double>(component(origin, kz));
    const auto point1_x =
        static_cast<double>(component(vertex1, kx)) - static_cast<double>(component(origin, kx));
    const auto point1_y =
        static_cast<double>(component(vertex1, ky)) - static_cast<double>(component(origin, ky));
    const auto point1_z =
        static_cast<double>(component(vertex1, kz)) - static_cast<double>(component(origin, kz));
    const auto point2_x =
        static_cast<double>(component(vertex2, kx)) - static_cast<double>(component(origin, kx));
    const auto point2_y =
        static_cast<double>(component(vertex2, ky)) - static_cast<double>(component(origin, ky));
    const auto point2_z =
        static_cast<double>(component(vertex2, kz)) - static_cast<double>(component(origin, kz));

    // Multiplying the usual sheared coordinates by direction_z avoids a
    // rounded reciprocal. Explicit round-to-nearest operations retain exact
    // antisymmetry when a shared edge is presented in reverse order.
    const auto ax = __dsub_rn(__dmul_rn(point0_x, direction_z), __dmul_rn(direction_x, point0_z));
    const auto ay = __dsub_rn(__dmul_rn(point0_y, direction_z), __dmul_rn(direction_y, point0_z));
    const auto bx = __dsub_rn(__dmul_rn(point1_x, direction_z), __dmul_rn(direction_x, point1_z));
    const auto by = __dsub_rn(__dmul_rn(point1_y, direction_z), __dmul_rn(direction_y, point1_z));
    const auto cx = __dsub_rn(__dmul_rn(point2_x, direction_z), __dmul_rn(direction_x, point2_z));
    const auto cy = __dsub_rn(__dmul_rn(point2_y, direction_z), __dmul_rn(direction_y, point2_z));
    const auto edge0 = __dsub_rn(__dmul_rn(bx, cy), __dmul_rn(by, cx));
    const auto edge1 = __dsub_rn(__dmul_rn(cx, ay), __dmul_rn(cy, ax));
    const auto edge2 = __dsub_rn(__dmul_rn(ax, by), __dmul_rn(ay, bx));
    if (!isfinite(edge0) || !isfinite(edge1) || !isfinite(edge2)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    const auto has_negative = edge0 < 0.0 || edge1 < 0.0 || edge2 < 0.0;
    const auto has_positive = edge0 > 0.0 || edge1 > 0.0 || edge2 > 0.0;
    if (has_negative && has_positive) {
        return SceneClosestHitStatus::miss;
    }

    const auto determinant = edge0 + edge1 + edge2;
    if (!isfinite(determinant)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    if (determinant == 0.0) {
        return SceneClosestHitStatus::miss;
    }
    const auto parameter_numerator = fma(edge0, point0_z, fma(edge1, point1_z, edge2 * point2_z));
    const auto parameter_denominator = determinant * direction_z;
    if (!isfinite(parameter_numerator) || !isfinite(parameter_denominator) ||
        parameter_denominator == 0.0) {
        return SceneClosestHitStatus::numerical_failure;
    }
    const auto parameter = parameter_numerator / parameter_denominator;
    const auto barycentric0 = edge0 / determinant;
    const auto barycentric1 = edge1 / determinant;
    const auto barycentric2 = edge2 / determinant;
    if (!isfinite(parameter) || !isfinite(barycentric0) || !isfinite(barycentric1) ||
        !isfinite(barycentric2)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    if (parameter < static_cast<double>(t_min) || parameter > static_cast<double>(t_max)) {
        return SceneClosestHitStatus::miss;
    }

    const auto float_parameter = static_cast<float>(parameter);
    const auto float_barycentric0 = static_cast<float>(barycentric0);
    const auto float_barycentric1 = static_cast<float>(barycentric1);
    const auto float_barycentric2 = static_cast<float>(barycentric2);
    if (!isfinite(float_parameter) || !isfinite(float_barycentric0) ||
        !isfinite(float_barycentric1) || !isfinite(float_barycentric2)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    result = TriangleIntersection{
        .hit = true,
        .parameter = float_parameter == 0.0F ? 0.0F : float_parameter,
        .barycentric0 = float_barycentric0 == 0.0F ? 0.0F : float_barycentric0,
        .barycentric1 = float_barycentric1 == 0.0F ? 0.0F : float_barycentric1,
        .barycentric2 = float_barycentric2 == 0.0F ? 0.0F : float_barycentric2,
    };
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ SceneClosestHitStatus
watertight_triangle(const Vector3 vertex0, const Vector3 vertex1, const Vector3 vertex2,
                    const Vector3 origin, const Vector3 direction, const float t_min,
                    const float t_max, TriangleIntersection& result) noexcept {
    auto kz = std::uint32_t{0U};
    auto maximum_direction = fabsf(direction.x);
    if (fabsf(direction.y) > maximum_direction) {
        kz = 1U;
        maximum_direction = fabsf(direction.y);
    }
    if (fabsf(direction.z) > maximum_direction) {
        kz = 2U;
    }
    auto kx = kz + 1U;
    if (kx == 3U) {
        kx = 0U;
    }
    auto ky = kx + 1U;
    if (ky == 3U) {
        ky = 0U;
    }
    if (component(direction, kz) < 0.0F) {
        const auto temporary = kx;
        kx = ky;
        ky = temporary;
    }

    const auto translated0 = subtract(vertex0, origin);
    const auto translated1 = subtract(vertex1, origin);
    const auto translated2 = subtract(vertex2, origin);
    if (!finite_vector(translated0) || !finite_vector(translated1) || !finite_vector(translated2)) {
        return SceneClosestHitStatus::numerical_failure;
    }

    const auto reciprocal_z = 1.0F / component(direction, kz);
    const auto shear_x = component(direction, kx) * reciprocal_z;
    const auto shear_y = component(direction, ky) * reciprocal_z;
    if (!isfinite(reciprocal_z) || !isfinite(shear_x) || !isfinite(shear_y)) {
        return SceneClosestHitStatus::numerical_failure;
    }

    const auto ax = fmaf(-shear_x, component(translated0, kz), component(translated0, kx));
    const auto ay = fmaf(-shear_y, component(translated0, kz), component(translated0, ky));
    const auto bx = fmaf(-shear_x, component(translated1, kz), component(translated1, kx));
    const auto by = fmaf(-shear_y, component(translated1, kz), component(translated1, ky));
    const auto cx = fmaf(-shear_x, component(translated2, kz), component(translated2, kx));
    const auto cy = fmaf(-shear_y, component(translated2, kz), component(translated2, ky));
    const auto az = component(translated0, kz) * reciprocal_z;
    const auto bz = component(translated1, kz) * reciprocal_z;
    const auto cz = component(translated2, kz) * reciprocal_z;
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(bx) || !isfinite(by) || !isfinite(cx) ||
        !isfinite(cy) || !isfinite(az) || !isfinite(bz) || !isfinite(cz)) {
        return SceneClosestHitStatus::numerical_failure;
    }

    // Separate round-to-nearest products and subtraction preserve the exact
    // sign reversal of a shared edge when adjacent triangles swap endpoints.
    auto edge0 = __fsub_rn(__fmul_rn(bx, cy), __fmul_rn(by, cx));
    auto edge1 = __fsub_rn(__fmul_rn(cx, ay), __fmul_rn(cy, ax));
    auto edge2 = __fsub_rn(__fmul_rn(ax, by), __fmul_rn(ay, bx));
    if (!isfinite(edge0) || !isfinite(edge1) || !isfinite(edge2)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    const auto has_negative = edge0 < 0.0F || edge1 < 0.0F || edge2 < 0.0F;
    const auto has_positive = edge0 > 0.0F || edge1 > 0.0F || edge2 > 0.0F;
    if (edge0 == 0.0F || edge1 == 0.0F || edge2 == 0.0F || (has_negative && has_positive)) {
        return refine_ambiguous_triangle_edges(vertex0, vertex1, vertex2, origin, direction, kx, ky,
                                               kz, t_min, t_max, result);
    }

    const auto determinant = edge0 + edge1 + edge2;
    if (!isfinite(determinant)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    if (determinant == 0.0F) {
        return SceneClosestHitStatus::miss;
    }
    const auto scaled_parameter = fmaf(edge0, az, fmaf(edge1, bz, edge2 * cz));
    if (!isfinite(scaled_parameter)) {
        return SceneClosestHitStatus::numerical_failure;
    }

    const auto scaled_minimum = t_min * determinant;
    const auto scaled_maximum = t_max * determinant;
    if (isnan(scaled_minimum) || isnan(scaled_maximum)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    if ((determinant > 0.0F &&
         (scaled_parameter < scaled_minimum || scaled_parameter > scaled_maximum)) ||
        (determinant < 0.0F &&
         (scaled_parameter > scaled_minimum || scaled_parameter < scaled_maximum))) {
        return SceneClosestHitStatus::miss;
    }

    const auto inverse_determinant = 1.0F / determinant;
    const auto parameter = scaled_parameter * inverse_determinant;
    const auto barycentric0 = edge0 * inverse_determinant;
    const auto barycentric1 = edge1 * inverse_determinant;
    const auto barycentric2 = edge2 * inverse_determinant;
    if (!isfinite(parameter) || !isfinite(barycentric0) || !isfinite(barycentric1) ||
        !isfinite(barycentric2)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    if (parameter < t_min || parameter > t_max) {
        return SceneClosestHitStatus::miss;
    }

    result = TriangleIntersection{
        .hit = true,
        .parameter = parameter == 0.0F ? 0.0F : parameter,
        .barycentric0 = barycentric0 == 0.0F ? 0.0F : barycentric0,
        .barycentric1 = barycentric1 == 0.0F ? 0.0F : barycentric1,
        .barycentric2 = barycentric2 == 0.0F ? 0.0F : barycentric2,
    };
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ SceneClosestHitStatus
world_geometric_normal(const std::uint8_t* const scene_bytes, const SceneSoaHeader& scene,
                       const std::uint32_t instance, const Vector3 vertex0, const Vector3 vertex1,
                       const Vector3 vertex2, Vector3& normal) noexcept {
    float matrix[shared::SceneSoaMatrixElementCount]{};
    auto status =
        load_matrix(scene_bytes, scene, scene_column::instance_local_to_world, instance, matrix);
    if (status != SceneClosestHitStatus::miss) {
        return status;
    }
    const auto world0 = transform_point(matrix, vertex0);
    const auto world1 = transform_point(matrix, vertex1);
    const auto world2 = transform_point(matrix, vertex2);
    if (!finite_vector(world0) || !finite_vector(world1) || !finite_vector(world2)) {
        return SceneClosestHitStatus::numerical_failure;
    }

    const auto first = subtract(world1, world0);
    const auto second = subtract(world2, world0);
    if (!finite_vector(first) || !finite_vector(second)) {
        return SceneClosestHitStatus::numerical_failure;
    }
    const auto cross_x = static_cast<double>(first.y) * static_cast<double>(second.z) -
                         static_cast<double>(first.z) * static_cast<double>(second.y);
    const auto cross_y = static_cast<double>(first.z) * static_cast<double>(second.x) -
                         static_cast<double>(first.x) * static_cast<double>(second.z);
    const auto cross_z = static_cast<double>(first.x) * static_cast<double>(second.y) -
                         static_cast<double>(first.y) * static_cast<double>(second.x);
    const auto maximum = fmax(fabs(cross_x), fmax(fabs(cross_y), fabs(cross_z)));
    if (!isfinite(cross_x) || !isfinite(cross_y) || !isfinite(cross_z) || !isfinite(maximum) ||
        maximum == 0.0) {
        return SceneClosestHitStatus::numerical_failure;
    }
    const auto scaled_x = cross_x / maximum;
    const auto scaled_y = cross_y / maximum;
    const auto scaled_z = cross_z / maximum;
    const auto length = sqrt(scaled_x * scaled_x + scaled_y * scaled_y + scaled_z * scaled_z);
    if (!isfinite(length) || length == 0.0) {
        return SceneClosestHitStatus::numerical_failure;
    }
    normal = Vector3{.x = static_cast<float>(scaled_x / length),
                     .y = static_cast<float>(scaled_y / length),
                     .z = static_cast<float>(scaled_z / length)};
    return finite_vector(normal) ? SceneClosestHitStatus::miss
                                 : SceneClosestHitStatus::numerical_failure;
}

[[nodiscard]] __device__ bool candidate_precedes(const float parameter,
                                                 const std::uint32_t instance_id,
                                                 const std::uint32_t primitive_id,
                                                 const ClosestCandidate& closest) noexcept {
    return !closest.present || parameter < closest.parameter ||
           (parameter == closest.parameter &&
            (instance_id < closest.instance_id ||
             (instance_id == closest.instance_id && primitive_id < closest.primitive_id)));
}

[[nodiscard]] __device__ SceneClosestHitStatus traverse_blas(
    const std::uint8_t* const scene_bytes, const SceneSoaHeader& scene,
    const std::uint8_t* const bvh_bytes, const SceneBvhHeader& bvh,
    const SceneBvhInstanceReference& instance_reference, const TransportRay& world_ray,
    const Vector3 local_origin, const Vector3 local_direction, ClosestCandidate& closest) noexcept {
    const auto* const blases = bvh_values<SceneBvhBlas>(bvh_bytes, bvh, bvh_array::blas);
    const auto* const nodes = bvh_values<SceneBvhNode>(bvh_bytes, bvh, bvh_array::blas_node);
    const auto* const primitive_references =
        bvh_values<std::uint32_t>(bvh_bytes, bvh, bvh_array::primitive_reference);
    if (instance_reference.blas_index >= bvh.blas_count ||
        instance_reference.scene_geometry_index >= scene.geometry_count) {
        return SceneClosestHitStatus::invalid_topology;
    }
    const auto& blas = blases[instance_reference.blas_index];
    const auto* const blas_reserved = reinterpret_cast<const std::uint32_t*>(
        reinterpret_cast<const std::uint8_t*>(&blas) + offsetof(SceneBvhBlas, reserved));
    const auto node_end = static_cast<std::uint64_t>(blas.node_offset) + blas.node_count;
    const auto reference_end = static_cast<std::uint64_t>(blas.primitive_reference_offset) +
                               blas.primitive_reference_count;
    if (blas_reserved[0] != 0U || blas_reserved[1] != 0U ||
        blas.geometry_id != instance_reference.geometry_id || blas.root_node != blas.node_offset ||
        blas.node_count == 0U || node_end > bvh.blas_node_count ||
        blas.node_count != static_cast<std::uint64_t>(blas.primitive_reference_count) * 2U - 1U ||
        reference_end > bvh.primitive_reference_count) {
        return SceneClosestHitStatus::invalid_topology;
    }

    const auto geometry_index = instance_reference.scene_geometry_index;
    const auto geometry_id =
        scene_values<std::uint32_t>(scene_bytes, scene, scene_column::geometry_id)[geometry_index];
    const auto vertex_offset = scene_values<std::uint64_t>(
        scene_bytes, scene, scene_column::geometry_vertex_offset)[geometry_index];
    const auto vertex_count = scene_values<std::uint64_t>(
        scene_bytes, scene, scene_column::geometry_vertex_count)[geometry_index];
    const auto triangle_offset = scene_values<std::uint64_t>(
        scene_bytes, scene, scene_column::geometry_triangle_offset)[geometry_index];
    const auto triangle_count = scene_values<std::uint64_t>(
        scene_bytes, scene, scene_column::geometry_triangle_count)[geometry_index];
    if (geometry_id != instance_reference.geometry_id ||
        triangle_count != blas.primitive_reference_count || vertex_count == 0U ||
        vertex_offset > scene.vertex_count || vertex_count > scene.vertex_count - vertex_offset ||
        triangle_offset > scene.triangle_count ||
        triangle_count > scene.triangle_count - triangle_offset) {
        return SceneClosestHitStatus::invalid_topology;
    }

    std::uint32_t stack[TraversalStackCapacity]{};
    auto stack_size = std::uint32_t{0U};
    if (!push_node(stack, stack_size, blas.root_node)) {
        return SceneClosestHitStatus::stack_overflow;
    }
    auto visited_node_count = std::uint32_t{0U};
    while (stack_size != 0U) {
        if (visited_node_count++ >= blas.node_count) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto node_index = stack[--stack_size];
        if (node_index < blas.node_offset || node_index >= node_end) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto& node = nodes[node_index];
        auto intersects = false;
        auto near_parameter = 0.0F;
        auto status = intersect_bounds(node, local_origin, local_direction, world_ray.t_min,
                                       closest.parameter, intersects, near_parameter);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
        if (!intersects) {
            continue;
        }
        if (node.kind == static_cast<std::uint32_t>(SceneBvhNodeKind::internal)) {
            if (node.split_axis > 2U || node.reference_offset != 0U || node.reference_count != 0U) {
                return SceneClosestHitStatus::invalid_topology;
            }
            status = push_intersecting_children(nodes, node.first_child, node.second_child,
                                                blas.node_offset, blas.node_count, local_origin,
                                                local_direction, world_ray.t_min, closest.parameter,
                                                stack, stack_size);
            if (status != SceneClosestHitStatus::miss) {
                return status;
            }
            continue;
        }
        if (node.kind != static_cast<std::uint32_t>(SceneBvhNodeKind::leaf) ||
            node.split_axis != shared::SceneBvhInvalidIndex ||
            node.first_child != shared::SceneBvhInvalidIndex ||
            node.second_child != shared::SceneBvhInvalidIndex || node.reference_count != 1U ||
            node.reference_offset < blas.primitive_reference_offset ||
            node.reference_offset >= reference_end) {
            return SceneClosestHitStatus::invalid_topology;
        }

        const auto primitive = primitive_references[node.reference_offset];
        if (primitive >= triangle_count) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto global_triangle = triangle_offset + primitive;
        Vector3 vertices[3]{};
        for (auto corner = std::uint32_t{0U}; corner < 3U; ++corner) {
            const auto local_vertex = scene_values<std::uint32_t>(
                scene_bytes, scene, scene_column::triangle_vertex_0 + corner)[global_triangle];
            if (local_vertex >= vertex_count) {
                return SceneClosestHitStatus::invalid_topology;
            }
            const auto global_vertex = vertex_offset + local_vertex;
            vertices[corner] = Vector3{
                .x = scene_values<float>(scene_bytes, scene,
                                         scene_column::position_x)[global_vertex],
                .y = scene_values<float>(scene_bytes, scene,
                                         scene_column::position_y)[global_vertex],
                .z = scene_values<float>(scene_bytes, scene,
                                         scene_column::position_z)[global_vertex],
            };
            if (!finite_vector(vertices[corner])) {
                return SceneClosestHitStatus::numerical_failure;
            }
        }

        auto intersection = TriangleIntersection{};
        status =
            watertight_triangle(vertices[0], vertices[1], vertices[2], local_origin,
                                local_direction, world_ray.t_min, closest.parameter, intersection);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
        if (!intersection.hit ||
            !candidate_precedes(intersection.parameter, instance_reference.instance_id, primitive,
                                closest)) {
            continue;
        }

        auto normal = Vector3{};
        status = world_geometric_normal(scene_bytes, scene, instance_reference.scene_instance_index,
                                        vertices[0], vertices[1], vertices[2], normal);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
        const auto scene_instance = instance_reference.scene_instance_index;
        const auto scene_instance_id = scene_values<std::uint32_t>(
            scene_bytes, scene, scene_column::instance_id)[scene_instance];
        const auto scene_geometry_id = scene_values<std::uint32_t>(
            scene_bytes, scene, scene_column::instance_geometry_id)[scene_instance];
        if (scene_instance_id != instance_reference.instance_id ||
            scene_geometry_id != instance_reference.geometry_id) {
            return SceneClosestHitStatus::invalid_topology;
        }

        closest.present = true;
        closest.parameter = intersection.parameter;
        closest.instance_id = instance_reference.instance_id;
        closest.primitive_id = primitive;
        closest.hit = ClosestHit{
            .parameter = intersection.parameter,
            .barycentric_vertex0 = intersection.barycentric0,
            .barycentric_vertex1 = intersection.barycentric1,
            .barycentric_vertex2 = intersection.barycentric2,
            .geometric_normal_x = normal.x,
            .geometric_normal_y = normal.y,
            .geometric_normal_z = normal.z,
            .reserved = 0U,
            .identifiers =
                {
                    .object = scene_values<std::uint32_t>(
                        scene_bytes, scene, scene_column::instance_object_id)[scene_instance],
                    .instance = instance_reference.instance_id,
                    .geometry = instance_reference.geometry_id,
                    .primitive = primitive,
                    .material = scene_values<std::uint32_t>(
                        scene_bytes, scene, scene_column::instance_material_id)[scene_instance],
                    .reserved = {0U, 0U, 0U},
                },
        };
    }
    return SceneClosestHitStatus::miss;
}

[[nodiscard]] __device__ SceneClosestHitStatus
trace_closest_hit(const std::uint8_t* const scene_bytes, const SceneSoaHeader& scene,
                  const std::uint8_t* const bvh_bytes, const SceneBvhHeader& bvh,
                  const TransportRay& ray, ClosestCandidate& closest) noexcept {
    if (bvh.tlas_node_count == 0U) {
        return SceneClosestHitStatus::miss;
    }
    const auto* const nodes = bvh_values<SceneBvhNode>(bvh_bytes, bvh, bvh_array::tlas_node);
    const auto* const references =
        bvh_values<SceneBvhInstanceReference>(bvh_bytes, bvh, bvh_array::instance_reference);
    std::uint32_t stack[TraversalStackCapacity]{};
    auto stack_size = std::uint32_t{0U};
    if (!push_node(stack, stack_size, bvh.tlas_root_node)) {
        return SceneClosestHitStatus::stack_overflow;
    }
    auto visited_node_count = std::uint32_t{0U};
    const auto world_origin = Vector3{.x = ray.origin_x, .y = ray.origin_y, .z = ray.origin_z};
    const auto world_direction =
        Vector3{.x = ray.direction_x, .y = ray.direction_y, .z = ray.direction_z};

    while (stack_size != 0U) {
        if (visited_node_count++ >= bvh.tlas_node_count) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto node_index = stack[--stack_size];
        if (node_index >= bvh.tlas_node_count) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto& node = nodes[node_index];
        auto intersects = false;
        auto near_parameter = 0.0F;
        auto status = intersect_bounds(node, world_origin, world_direction, ray.t_min,
                                       closest.parameter, intersects, near_parameter);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
        if (!intersects) {
            continue;
        }
        if (node.kind == static_cast<std::uint32_t>(SceneBvhNodeKind::internal)) {
            if (node.split_axis > 2U || node.reference_offset != 0U || node.reference_count != 0U) {
                return SceneClosestHitStatus::invalid_topology;
            }
            status = push_intersecting_children(nodes, node.first_child, node.second_child, 0U,
                                                static_cast<std::uint32_t>(bvh.tlas_node_count),
                                                world_origin, world_direction, ray.t_min,
                                                closest.parameter, stack, stack_size);
            if (status != SceneClosestHitStatus::miss) {
                return status;
            }
            continue;
        }
        if (node.kind != static_cast<std::uint32_t>(SceneBvhNodeKind::leaf) ||
            node.split_axis != shared::SceneBvhInvalidIndex ||
            node.first_child != shared::SceneBvhInvalidIndex ||
            node.second_child != shared::SceneBvhInvalidIndex || node.reference_count != 1U ||
            node.reference_offset >= bvh.instance_reference_count) {
            return SceneClosestHitStatus::invalid_topology;
        }

        const auto& reference = references[node.reference_offset];
        const auto* const reference_reserved = reinterpret_cast<const std::uint32_t*>(
            reinterpret_cast<const std::uint8_t*>(&reference) +
            offsetof(SceneBvhInstanceReference, reserved));
        if (reference_reserved[0] != 0U || reference_reserved[1] != 0U ||
            reference.scene_instance_index >= scene.instance_count ||
            reference.scene_geometry_index >= scene.geometry_count ||
            reference.blas_index >= bvh.blas_count) {
            return SceneClosestHitStatus::invalid_topology;
        }
        const auto scene_instance = reference.scene_instance_index;
        const auto scene_instance_id = scene_values<std::uint32_t>(
            scene_bytes, scene, scene_column::instance_id)[scene_instance];
        const auto scene_geometry_id = scene_values<std::uint32_t>(
            scene_bytes, scene, scene_column::instance_geometry_id)[scene_instance];
        const auto scene_visibility = scene_values<std::uint32_t>(
            scene_bytes, scene, scene_column::instance_visibility_mask)[scene_instance];
        if (scene_instance_id != reference.instance_id ||
            scene_geometry_id != reference.geometry_id ||
            scene_visibility != reference.visibility_mask) {
            return SceneClosestHitStatus::invalid_topology;
        }
        if ((ray.visibility_mask & reference.visibility_mask) == 0U) {
            continue;
        }

        auto local_origin = Vector3{};
        auto local_direction = Vector3{};
        status = object_space_ray(scene_bytes, scene, scene_instance, ray, local_origin,
                                  local_direction);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
        status = traverse_blas(scene_bytes, scene, bvh_bytes, bvh, reference, ray, local_origin,
                               local_direction, closest);
        if (status != SceneClosestHitStatus::miss) {
            return status;
        }
    }
    return SceneClosestHitStatus::miss;
}

__global__ void scene_closest_hit_kernel(const std::uint8_t* const scene_bytes,
                                         const std::size_t scene_size,
                                         const std::uint8_t* const bvh_bytes,
                                         const std::size_t bvh_size, const TransportRay* const rays,
                                         const std::uint32_t ray_count,
                                         SceneClosestHitResult* const results) {
    __shared__ std::uint32_t layout_status;
    if (threadIdx.x == 0U) {
        auto status = validate_scene_layout(scene_bytes, scene_size);
        if (status == SceneClosestHitStatus::miss) {
            status = validate_bvh_layout(bvh_bytes, bvh_size,
                                         *reinterpret_cast<const SceneSoaHeader*>(scene_bytes));
        }
        layout_status = status_value(status);
    }
    __syncthreads();

    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= ray_count) {
        return;
    }
    auto output = SceneClosestHitResult{};
    if (layout_status != status_value(SceneClosestHitStatus::miss)) {
        output.status = layout_status;
        results[index] = output;
        return;
    }

    const auto ray = rays[index];
    if (!valid_ray(ray)) {
        output.status = status_value(SceneClosestHitStatus::invalid_ray);
        results[index] = output;
        return;
    }
    auto closest = ClosestCandidate{.parameter = ray.t_max};
    const auto status = trace_closest_hit(
        scene_bytes, *reinterpret_cast<const SceneSoaHeader*>(scene_bytes), bvh_bytes,
        *reinterpret_cast<const SceneBvhHeader*>(bvh_bytes), ray, closest);
    if (status != SceneClosestHitStatus::miss) {
        output.status = status_value(status);
    } else if (closest.present) {
        output.status = status_value(SceneClosestHitStatus::hit);
        output.hit = closest.hit;
    }
    results[index] = output;
}

} // namespace

extern "C" int blackframe_cuda_launch_scene_closest_hit(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const std::uint8_t* const bvh_bytes, const std::size_t bvh_size,
    const blackframe::xpu::shared::TransportRay* const rays, const std::uint32_t ray_count,
    blackframe::xpu::shared::SceneClosestHitResult* const results) noexcept {
    if (scene_bytes == nullptr || scene_size < sizeof(blackframe::xpu::shared::SceneSoaHeader) ||
        bvh_bytes == nullptr || bvh_size < sizeof(blackframe::xpu::shared::SceneBvhHeader) ||
        (ray_count != 0U && (rays == nullptr || results == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (ray_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }

    const auto block_count = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(ray_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
    scene_closest_hit_kernel<<<block_count, ThreadsPerBlock>>>(scene_bytes, scene_size, bvh_bytes,
                                                               bvh_size, rays, ray_count, results);
    return static_cast<int>(cudaGetLastError());
}
