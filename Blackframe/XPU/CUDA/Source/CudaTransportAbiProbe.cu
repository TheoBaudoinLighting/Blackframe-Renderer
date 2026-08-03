#include <Blackframe/XPU/CUDA/TransportAbiProbe.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The host/device ABI probe must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in host/device ABI code."
#endif

namespace {

using blackframe::xpu::shared::ClosestHit;
using blackframe::xpu::shared::HostDeviceLayoutValueCount;
using blackframe::xpu::shared::HostDeviceTransportAbiMajor;
using blackframe::xpu::shared::HostDeviceTransportAbiMinor;
using blackframe::xpu::shared::LayoutManifest;
using blackframe::xpu::shared::LayoutValue;
using blackframe::xpu::shared::PathSlot;
using blackframe::xpu::shared::QueueHeader;
using blackframe::xpu::shared::SampleStreamIndex;
using blackframe::xpu::shared::SurfaceIdentifiers;
using blackframe::xpu::shared::TransportPathStateLane;
using blackframe::xpu::shared::TransportRay;
using blackframe::xpu::shared::TransportSpectrum;

__global__ void query_transport_abi_layout_kernel(LayoutManifest* const output) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }

    auto manifest = LayoutManifest{};
    manifest.abi_major = HostDeviceTransportAbiMajor;
    manifest.abi_minor = HostDeviceTransportAbiMinor;
    manifest.device_cxx_standard = __cplusplus;
    manifest.value_count = HostDeviceLayoutValueCount;

#define BLACKFRAME_SET_LAYOUT_VALUE(name, expression)                                              \
    manifest.values[static_cast<std::uint32_t>(LayoutValue::name)] =                               \
        static_cast<std::uint32_t>(expression)

    BLACKFRAME_SET_LAYOUT_VALUE(spectrum_size, sizeof(TransportSpectrum));
    BLACKFRAME_SET_LAYOUT_VALUE(spectrum_alignment, alignof(TransportSpectrum));
    BLACKFRAME_SET_LAYOUT_VALUE(spectrum_values_offset, offsetof(TransportSpectrum, values));

    BLACKFRAME_SET_LAYOUT_VALUE(ray_size, sizeof(TransportRay));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_alignment, alignof(TransportRay));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_origin_x_offset, offsetof(TransportRay, origin_x));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_origin_y_offset, offsetof(TransportRay, origin_y));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_origin_z_offset, offsetof(TransportRay, origin_z));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_t_min_offset, offsetof(TransportRay, t_min));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_direction_x_offset, offsetof(TransportRay, direction_x));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_direction_y_offset, offsetof(TransportRay, direction_y));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_direction_z_offset, offsetof(TransportRay, direction_z));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_t_max_offset, offsetof(TransportRay, t_max));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_time_offset, offsetof(TransportRay, time));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_visibility_mask_offset,
                                offsetof(TransportRay, visibility_mask));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_current_medium_offset, offsetof(TransportRay, current_medium));
    BLACKFRAME_SET_LAYOUT_VALUE(ray_reserved_offset, offsetof(TransportRay, reserved));

    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_size, sizeof(SampleStreamIndex));
    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_alignment, alignof(SampleStreamIndex));
    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_pixel_x_offset,
                                offsetof(SampleStreamIndex, pixel_x));
    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_pixel_y_offset,
                                offsetof(SampleStreamIndex, pixel_y));
    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_sample_index_offset,
                                offsetof(SampleStreamIndex, sample_index));
    BLACKFRAME_SET_LAYOUT_VALUE(sample_stream_index_seed_offset, offsetof(SampleStreamIndex, seed));

    BLACKFRAME_SET_LAYOUT_VALUE(path_slot_size, sizeof(PathSlot));
    BLACKFRAME_SET_LAYOUT_VALUE(path_slot_alignment, alignof(PathSlot));
    BLACKFRAME_SET_LAYOUT_VALUE(path_slot_value_offset, offsetof(PathSlot, value));

    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_size, sizeof(QueueHeader));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_alignment, alignof(QueueHeader));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_abi_major_offset, offsetof(QueueHeader, abi_major));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_abi_minor_offset, offsetof(QueueHeader, abi_minor));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_struct_size_offset,
                                offsetof(QueueHeader, struct_size));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_kind_offset, offsetof(QueueHeader, queue_kind));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_capacity_offset, offsetof(QueueHeader, capacity));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_size_offset, offsetof(QueueHeader, size));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_overflow_count_offset,
                                offsetof(QueueHeader, overflow_count));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_rejected_count_offset,
                                offsetof(QueueHeader, rejected_count));
    BLACKFRAME_SET_LAYOUT_VALUE(queue_header_reserved_offset, offsetof(QueueHeader, reserved));

    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_size, sizeof(SurfaceIdentifiers));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_alignment, alignof(SurfaceIdentifiers));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_object_offset,
                                offsetof(SurfaceIdentifiers, object));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_instance_offset,
                                offsetof(SurfaceIdentifiers, instance));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_geometry_offset,
                                offsetof(SurfaceIdentifiers, geometry));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_primitive_offset,
                                offsetof(SurfaceIdentifiers, primitive));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_material_offset,
                                offsetof(SurfaceIdentifiers, material));
    BLACKFRAME_SET_LAYOUT_VALUE(surface_identifiers_reserved_offset,
                                offsetof(SurfaceIdentifiers, reserved));

    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_size, sizeof(ClosestHit));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_alignment, alignof(ClosestHit));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_parameter_offset, offsetof(ClosestHit, parameter));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_barycentric_vertex0_offset,
                                offsetof(ClosestHit, barycentric_vertex0));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_barycentric_vertex1_offset,
                                offsetof(ClosestHit, barycentric_vertex1));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_barycentric_vertex2_offset,
                                offsetof(ClosestHit, barycentric_vertex2));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_geometric_normal_x_offset,
                                offsetof(ClosestHit, geometric_normal_x));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_geometric_normal_y_offset,
                                offsetof(ClosestHit, geometric_normal_y));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_geometric_normal_z_offset,
                                offsetof(ClosestHit, geometric_normal_z));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_reserved_offset, offsetof(ClosestHit, reserved));
    BLACKFRAME_SET_LAYOUT_VALUE(closest_hit_identifiers_offset, offsetof(ClosestHit, identifiers));

    BLACKFRAME_SET_LAYOUT_VALUE(path_state_size, sizeof(TransportPathStateLane));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_alignment, alignof(TransportPathStateLane));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_beta_offset, offsetof(TransportPathStateLane, beta));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_accumulated_radiance_offset,
                                offsetof(TransportPathStateLane, accumulated_radiance));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_wavelength_nanometers_offset,
                                offsetof(TransportPathStateLane, wavelength_nanometers));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_wavelength_pdf_values_offset,
                                offsetof(TransportPathStateLane, wavelength_pdf_values));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_diffuse_depth_offset,
                                offsetof(TransportPathStateLane, diffuse_depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_glossy_depth_offset,
                                offsetof(TransportPathStateLane, glossy_depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_specular_depth_offset,
                                offsetof(TransportPathStateLane, specular_depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_transmission_depth_offset,
                                offsetof(TransportPathStateLane, transmission_depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_volume_depth_offset,
                                offsetof(TransportPathStateLane, volume_depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_depth_offset, offsetof(TransportPathStateLane, depth));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_eta_scale_offset,
                                offsetof(TransportPathStateLane, eta_scale));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_current_medium_offset,
                                offsetof(TransportPathStateLane, current_medium));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_delta_flags_offset,
                                offsetof(TransportPathStateLane, delta_flags));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_wavelength_pdf_measures_offset,
                                offsetof(TransportPathStateLane, wavelength_pdf_measures));
    BLACKFRAME_SET_LAYOUT_VALUE(path_state_reserved_offset,
                                offsetof(TransportPathStateLane, reserved));

#undef BLACKFRAME_SET_LAYOUT_VALUE

    *output = manifest;
}

} // namespace

extern "C" int blackframe_cuda_query_transport_abi_layout(LayoutManifest* const output,
                                                          int* const device_count) noexcept {
    if (output == nullptr || device_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *output = LayoutManifest{};
    *device_count = 0;

    auto status = cudaGetDeviceCount(device_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (*device_count == 0) {
        return static_cast<int>(cudaErrorNoDevice);
    }

    status = cudaSetDevice(0);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    LayoutManifest* device_manifest = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&device_manifest), sizeof(LayoutManifest));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    query_transport_abi_layout_kernel<<<1, 1>>>(device_manifest);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status =
            cudaMemcpy(output, device_manifest, sizeof(LayoutManifest), cudaMemcpyDeviceToHost);
    }

    const auto free_status = cudaFree(device_manifest);
    if (status == cudaSuccess) {
        status = free_status;
    }
    return static_cast<int>(status);
}
