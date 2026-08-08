#include <Blackframe/XPU/CUDA/ConstantTextureKernel.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

namespace scene_column = blackframe::xpu::shared::scene_soa_column;
namespace shared = blackframe::xpu::shared;

using shared::ConstantTextureEvaluationRequest;
using shared::ConstantTextureEvaluationResult;
using shared::ConstantTextureEvaluationStatus;
using shared::ConstantTextureKind;
using shared::SceneSoaColumnDescriptor;
using shared::SceneSoaHeader;

inline constexpr auto ThreadsPerBlock = std::uint32_t{128U};
inline constexpr auto MaximumU64 = ~std::uint64_t{0U};

[[nodiscard]] __device__ constexpr std::uint64_t align_up(const std::uint64_t input,
                                                          const std::uint64_t alignment) noexcept {
    const auto remainder = input % alignment;
    return remainder == 0U ? input : input + (alignment - remainder);
}

[[nodiscard]] __device__ const SceneSoaColumnDescriptor&
scene_descriptor(const SceneSoaHeader* const header, const std::uint32_t column) noexcept {
    const auto* const descriptors = reinterpret_cast<const SceneSoaColumnDescriptor*>(
        reinterpret_cast<const std::uint8_t*>(header) + offsetof(SceneSoaHeader, columns));
    return descriptors[column];
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
    if (column >= scene_column::material_id && column < scene_column::closure_kind) {
        return header.material_count;
    }
    if (column >= scene_column::closure_kind && column < scene_column::instance_id) {
        return header.closure_count;
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
    if (column >= scene_column::environment_wavelength_nanometers &&
        column < scene_column::texture_id) {
        return header.environment_count;
    }
    if (column >= scene_column::texture_id && column < scene_column::material_normal_map_present) {
        return header.texture_count;
    }
    if (column >= scene_column::material_normal_map_present &&
        column < scene_column::image_texture_id) {
        return header.material_count;
    }
    if (column >= scene_column::image_texture_id && column < scene_column::image_mip_width) {
        return header.image_texture_count;
    }
    if (column >= scene_column::image_mip_width && column < scene_column::image_texel_value) {
        return header.image_mip_count;
    }
    if (column == scene_column::image_texel_value) {
        return header.image_texel_count;
    }
    return 0U;
}

[[nodiscard]] __device__ std::uint32_t
scene_column_element_size(const std::uint32_t column) noexcept {
    if (column >= scene_column::count) {
        return 0U;
    }
    if ((column >= scene_column::geometry_vertex_offset &&
         column <= scene_column::geometry_triangle_count) ||
        column == scene_column::material_closure_offset ||
        column == scene_column::material_closure_count) {
        return sizeof(std::uint64_t);
    }
    if (column == scene_column::material_spectral_present ||
        (column >= scene_column::material_wavelength_measure &&
         column < scene_column::material_closure_offset) ||
        column == scene_column::material_closure_frame_mode ||
        column == scene_column::instance_parent_present ||
        column == scene_column::material_normal_map_present ||
        column == scene_column::material_bump_map_present ||
        (column >= scene_column::environment_wavelength_measure &&
         column < scene_column::environment_radiance)) {
        return sizeof(std::uint8_t);
    }
    if ((column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) ||
        (column >= scene_column::material_wavelength_nanometers &&
         column < scene_column::material_wavelength_measure) ||
        column == scene_column::material_closure_tangent_rotation_radians ||
        (column >= scene_column::material_emitted_radiance &&
         column < scene_column::closure_kind) ||
        (column >= scene_column::closure_weight && column < scene_column::instance_id) ||
        (column >= scene_column::instance_local_to_parent &&
         column < scene_column::punctual_kind) ||
        (column >= scene_column::punctual_position_x &&
         column < scene_column::mesh_area_light_instance_id) ||
        (column >= scene_column::environment_wavelength_nanometers &&
         column < scene_column::environment_wavelength_measure) ||
        (column >= scene_column::environment_radiance && column < scene_column::texture_id) ||
        (column >= scene_column::texture_value &&
         column < scene_column::material_normal_map_present) ||
        column == scene_column::material_bump_map_scale ||
        column == scene_column::image_texel_value) {
        return sizeof(float);
    }
    if (column == scene_column::image_texture_mip_offset ||
        column == scene_column::image_texture_mip_count ||
        column == scene_column::image_mip_texel_offset ||
        column == scene_column::image_mip_texel_count) {
        return sizeof(std::uint64_t);
    }
    return sizeof(std::uint32_t);
}

[[nodiscard]] __device__ bool valid_scene(const std::uint8_t* const bytes,
                                          const std::size_t byte_count) noexcept {
    if (bytes == nullptr || byte_count < sizeof(SceneSoaHeader)) {
        return false;
    }
    const auto* const header = reinterpret_cast<const SceneSoaHeader*>(bytes);
    if (header->magic != shared::SceneSoaMagic || header->abi_major != shared::SceneSoaAbiMajor ||
        header->abi_minor != shared::SceneSoaAbiMinor ||
        header->header_size != sizeof(SceneSoaHeader) ||
        header->column_count != scene_column::count ||
        header->hash_algorithm != shared::SceneSoaHashAlgorithmFnv1a64 ||
        header->environment_count > 1U || header->total_size_bytes != byte_count ||
        header->total_size_bytes < sizeof(SceneSoaHeader) ||
        header->texture_count > 0xFFFFFFFFULL || header->image_texture_count > 0xFFFFFFFFULL ||
        header->image_mip_count > 0xFFFFFFFFULL) {
        return false;
    }
    const auto* const reserved =
        reinterpret_cast<const std::uint64_t*>(bytes + offsetof(SceneSoaHeader, reserved));
    for (auto index = std::uint32_t{0U}; index < 5U; ++index) {
        if (reserved[index] != 0U) {
            return false;
        }
    }
    auto cursor = align_up(sizeof(SceneSoaHeader), shared::SceneSoaColumnAlignment);
    for (auto column = std::uint32_t{0U}; column < scene_column::count; ++column) {
        const auto& descriptor = scene_descriptor(header, column);
        const auto expected_count = scene_column_count(*header, column);
        const auto expected_size = scene_column_element_size(column);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return false;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return false;
            }
            continue;
        }
        if (cursor > MaximumU64 - (shared::SceneSoaColumnAlignment - 1U)) {
            return false;
        }
        cursor = align_up(cursor, shared::SceneSoaColumnAlignment);
        if (descriptor.offset_bytes != cursor || expected_count > MaximumU64 / expected_size) {
            return false;
        }
        const auto column_bytes = expected_count * expected_size;
        if (cursor > MaximumU64 - column_bytes) {
            return false;
        }
        cursor += column_bytes;
    }
    return cursor == header->total_size_bytes;
}

template <class Element>
[[nodiscard]] __device__ const Element* scene_values(const std::uint8_t* const bytes,
                                                     const SceneSoaHeader& header,
                                                     const std::uint32_t column) noexcept {
    return reinterpret_cast<const Element*>(bytes + scene_descriptor(&header, column).offset_bytes);
}

[[nodiscard]] __device__ constexpr bool known_kind(const ConstantTextureKind kind) noexcept {
    return kind == ConstantTextureKind::float_value || kind == ConstantTextureKind::linear_rgb ||
           kind == ConstantTextureKind::sampled_spectrum;
}

[[nodiscard]] __device__ bool valid_record(const ConstantTextureKind kind,
                                           const float (&values)[4U]) noexcept {
    if (!known_kind(kind)) {
        return false;
    }
    for (auto lane = std::uint32_t{0U}; lane < shared::ConstantTextureValueCount; ++lane) {
        if (!isfinite(values[lane])) {
            return false;
        }
    }
    if (kind == ConstantTextureKind::float_value) {
        return __float_as_uint(values[1U]) == 0U && __float_as_uint(values[2U]) == 0U &&
               __float_as_uint(values[3U]) == 0U;
    }
    if (kind == ConstantTextureKind::linear_rgb) {
        return __float_as_uint(values[3U]) == 0U;
    }
    return true;
}

__global__ void evaluate_constant_textures_kernel(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const ConstantTextureEvaluationRequest* const requests, const std::uint32_t request_count,
    ConstantTextureEvaluationResult* const results) {
    const auto lane = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= request_count) {
        return;
    }

    auto output = ConstantTextureEvaluationResult{};
    if (!valid_scene(scene_bytes, scene_size)) {
        output.status = ConstantTextureEvaluationStatus::invalid_scene;
        results[lane] = output;
        return;
    }
    const auto request = requests[lane];
    if (!known_kind(request.expected_kind) || request.reserved[0U] != 0U ||
        request.reserved[1U] != 0U) {
        output.status = ConstantTextureEvaluationStatus::invalid_request;
        results[lane] = output;
        return;
    }

    const auto& scene = *reinterpret_cast<const SceneSoaHeader*>(scene_bytes);
    const auto* const ids =
        scene_values<std::uint32_t>(scene_bytes, scene, scene_column::texture_id);
    const auto* const kinds =
        scene_values<std::uint32_t>(scene_bytes, scene, scene_column::texture_kind);
    auto found = false;
    auto found_kind = ConstantTextureKind::float_value;
    float found_values[shared::ConstantTextureValueCount]{};
    auto previous_id = std::uint32_t{};
    for (auto texture = std::uint32_t{0U}; texture < scene.texture_count; ++texture) {
        const auto id = ids[texture];
        if (texture != 0U && id <= previous_id) {
            output.status = ConstantTextureEvaluationStatus::invalid_record;
            results[lane] = output;
            return;
        }
        previous_id = id;
        const auto kind = static_cast<ConstantTextureKind>(kinds[texture]);
        float values[shared::ConstantTextureValueCount]{};
        for (auto value = std::uint32_t{0U}; value < shared::ConstantTextureValueCount; ++value) {
            values[value] = scene_values<float>(scene_bytes, scene,
                                                scene_column::texture_value + value)[texture];
        }
        if (!valid_record(kind, values)) {
            output.status = ConstantTextureEvaluationStatus::invalid_record;
            results[lane] = output;
            return;
        }
        if (id == request.texture_id) {
            found = true;
            found_kind = kind;
            for (auto value = std::uint32_t{0U}; value < shared::ConstantTextureValueCount;
                 ++value) {
                found_values[value] = values[value];
            }
        }
    }
    if (!found) {
        output.status = ConstantTextureEvaluationStatus::unknown_texture;
        results[lane] = output;
        return;
    }
    if (found_kind != request.expected_kind) {
        output.status = ConstantTextureEvaluationStatus::type_mismatch;
        results[lane] = output;
        return;
    }
    for (auto value = std::uint32_t{0U}; value < shared::ConstantTextureValueCount; ++value) {
        output.values[value] = found_values[value];
    }
    output.kind = found_kind;
    output.status = ConstantTextureEvaluationStatus::success;
    results[lane] = output;
}

} // namespace

extern "C" int blackframe_cuda_launch_constant_texture_evaluation(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const blackframe::xpu::shared::ConstantTextureEvaluationRequest* const requests,
    const std::uint32_t request_count,
    blackframe::xpu::shared::ConstantTextureEvaluationResult* const results,
    void* const stream_handle) noexcept {
    if (scene_bytes == nullptr || scene_size < sizeof(blackframe::xpu::shared::SceneSoaHeader) ||
        (request_count != 0U && (requests == nullptr || results == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (request_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }

    const auto block_count = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(request_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    evaluate_constant_textures_kernel<<<block_count, ThreadsPerBlock, 0U, stream>>>(
        scene_bytes, scene_size, requests, request_count, results);
    return static_cast<int>(cudaGetLastError());
}
