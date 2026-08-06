#include <Blackframe/XPU/CUDA/TransportLobeProbe.hpp>
#include <Blackframe/XPU/CUDA/TransportLobesDevice.cuh>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The transport-lobe probe must be compiled by the CUDA compiler."
#endif

namespace {

namespace device = blackframe::xpu::cuda::transport_device;
namespace shared = blackframe::xpu::shared;
using blackframe::xpu::cuda::TransportLobeProbeResult;

[[nodiscard]] __device__ shared::TransportSpectrum spectrum(const float value) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < device::SpectrumLaneCount; ++lane) {
        result.values[lane] = value;
    }
    return result;
}

[[nodiscard]] __device__ bool positive(const shared::TransportSpectrum& value) noexcept {
    return device::spectrum_is_finite_nonnegative(value) && !device::spectrum_is_zero(value);
}

__global__ void transport_lobe_probe_kernel(TransportLobeProbeResult* const output) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    auto result = TransportLobeProbeResult{};
    result.device_cxx_standard = __cplusplus;
    const auto normal = device::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F};
    const auto coefficient = spectrum(0.5F);

    const auto lambert_eval = device::lambert_eval(coefficient, normal, normal);
    const auto lambert_pdf = device::lambert_pdf(normal, normal);
    const auto lambert_sample = device::sample_lambert(coefficient, normal, 0.25F, 0.75F);
    if (device::succeeded(lambert_eval.status) && device::succeeded(lambert_pdf.status) &&
        device::succeeded(lambert_sample.status) && positive(lambert_eval.value) &&
        lambert_pdf.value.value > 0.0F && lambert_sample.probability.value > 0.0F) {
        result.passed_mask |= 1U << 0U;
    }
    result.representative_values[0U] = lambert_pdf.value.value;

    const auto diffuse_eval = device::rough_diffuse_eval(coefficient, 0.6F, normal, normal);
    const auto diffuse_pdf = device::rough_diffuse_pdf(0.6F, normal, normal);
    const auto diffuse_sample =
        device::sample_rough_diffuse(coefficient, 0.6F, normal, 0.25F, 0.75F);
    if (device::succeeded(diffuse_eval.status) && device::succeeded(diffuse_pdf.status) &&
        device::succeeded(diffuse_sample.status) && positive(diffuse_eval.value) &&
        diffuse_pdf.value.value > 0.0F && diffuse_sample.probability.value > 0.0F) {
        result.passed_mask |= 1U << 1U;
    }
    result.representative_values[1U] = diffuse_eval.value.values[0U];

    const auto visible = device::ggx_sample_visible_normal(0.35F, 0.6F, normal, 0.2F, 0.7F);
    if (device::succeeded(visible.status) && device::unit_vector(visible.microfacet_normal) &&
        visible.probability.value > 0.0F) {
        result.passed_mask |= 1U << 2U;
    }
    result.representative_values[2U] = visible.probability.value;

    const auto eta = spectrum(0.2F);
    const auto k = spectrum(3.0F);
    const auto conductor_eval =
        device::rough_conductor_eval(coefficient, eta, k, 0.35F, 0.6F, normal, normal);
    const auto conductor_pdf = device::rough_conductor_pdf(0.35F, 0.6F, normal, normal);
    const auto conductor_sample =
        device::sample_rough_conductor(coefficient, eta, k, 0.35F, 0.6F, normal, 0.2F, 0.7F);
    if (device::succeeded(conductor_eval.status) && device::succeeded(conductor_pdf.status) &&
        device::succeeded(conductor_sample.status) && positive(conductor_eval.value) &&
        conductor_pdf.value.value > 0.0F && conductor_sample.probability.value > 0.0F) {
        result.passed_mask |= 1U << 3U;
    }
    result.representative_values[3U] = conductor_eval.value.values[0U];

    const auto dielectric_reflection = device::rough_dielectric_eval(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, normal, device::TransportMode::radiance);
    const auto dielectric_reflection_pdf = device::rough_dielectric_pdf(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, normal, device::TransportMode::radiance);
    if (device::succeeded(dielectric_reflection.status) &&
        device::succeeded(dielectric_reflection_pdf.status) &&
        positive(dielectric_reflection.value) && dielectric_reflection_pdf.value.value > 0.0F) {
        result.passed_mask |= 1U << 4U;
    }
    result.representative_values[4U] = dielectric_reflection_pdf.value.value;

    const auto below = device::Vector3{.x = 0.0F, .y = 0.0F, .z = -1.0F};
    const auto dielectric_transmission = device::rough_dielectric_eval(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, below, device::TransportMode::radiance);
    const auto dielectric_transmission_pdf = device::rough_dielectric_pdf(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, below, device::TransportMode::radiance);
    const auto dielectric_sample =
        device::sample_rough_dielectric(coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, 0.5F, 0.0F,
                                        0.0F, device::TransportMode::radiance);
    if (device::succeeded(dielectric_transmission.status) &&
        device::succeeded(dielectric_transmission_pdf.status) &&
        device::succeeded(dielectric_sample.status) && positive(dielectric_transmission.value) &&
        dielectric_transmission_pdf.value.value > 0.0F &&
        dielectric_sample.incoming_local.z < 0.0F) {
        result.passed_mask |= 1U << 5U;
    }
    result.representative_values[5U] = dielectric_transmission.value.values[0U];

    const auto conditional_reflection = device::rough_dielectric_pdf_with_direction_mask(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, normal, device::TransportMode::radiance,
        device::ScatteringLobe::reflection);
    const auto conditional_transmission = device::rough_dielectric_pdf_with_direction_mask(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, below, device::TransportMode::radiance,
        device::ScatteringLobe::transmission);
    const auto removed_transmission = device::rough_dielectric_eval_with_direction_mask(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, below, device::TransportMode::radiance,
        device::ScatteringLobe::reflection);
    const auto forced_transmission = device::sample_rough_dielectric_with_direction_mask(
        coefficient, 1.0F, 1.5F, 0.4F, 0.4F, normal, 0.0F, 0.0F, 0.0F,
        device::TransportMode::radiance, device::ScatteringLobe::transmission);
    const auto inside = device::Vector3{.x = 0.8F, .y = 0.0F, .z = -0.6F};
    auto rejected_tir_without_reflection_fallback = false;
    for (auto sample_y = std::uint32_t{};
         sample_y < 16U && !rejected_tir_without_reflection_fallback; ++sample_y) {
        for (auto sample_x = std::uint32_t{};
             sample_x < 16U && !rejected_tir_without_reflection_fallback; ++sample_x) {
            const auto canonical_u = (static_cast<float>(sample_x) + 0.5F) / 16.0F;
            const auto canonical_v = (static_cast<float>(sample_y) + 0.5F) / 16.0F;
            const auto unfiltered = device::sample_rough_dielectric(
                coefficient, 1.0F, 1.5F, 0.05F, 0.2F, inside, 0x1.fffffep-1F, canonical_u,
                canonical_v, device::TransportMode::radiance);
            if (!device::succeeded(unfiltered.status) ||
                !device::has_scattering_lobe(unfiltered.lobes,
                                             device::ScatteringLobe::reflection)) {
                continue;
            }
            const auto filtered = device::sample_rough_dielectric_with_direction_mask(
                coefficient, 1.0F, 1.5F, 0.05F, 0.2F, inside, 0.5F, canonical_u, canonical_v,
                device::TransportMode::radiance, device::ScatteringLobe::transmission);
            rejected_tir_without_reflection_fallback =
                filtered.status == device::Status::outside_support &&
                filtered.lobes == device::ScatteringLobe::none;
        }
    }
    if (device::succeeded(conditional_reflection.status) &&
        device::succeeded(conditional_transmission.status) &&
        conditional_reflection.value.value > dielectric_reflection_pdf.value.value &&
        conditional_transmission.value.value > dielectric_transmission_pdf.value.value &&
        device::succeeded(removed_transmission.status) &&
        device::spectrum_is_zero(removed_transmission.value) &&
        device::succeeded(forced_transmission.status) &&
        device::has_scattering_lobe(forced_transmission.lobes,
                                    device::ScatteringLobe::transmission) &&
        forced_transmission.incoming_local.z < 0.0F && rejected_tir_without_reflection_fallback) {
        result.passed_mask |= 1U << 10U;
    }
    result.representative_values[10U] = conditional_reflection.value.value;

    const auto mirror_eval = device::specular_reflection_eval(coefficient, normal, normal);
    const auto mirror_pdf = device::specular_reflection_pdf(coefficient, normal, normal);
    const auto mirror = device::sample_specular_reflection(coefficient, normal);
    if (device::succeeded(mirror_eval.status) && device::succeeded(mirror_pdf.status) &&
        device::spectrum_is_zero(mirror_eval.value) && mirror_pdf.value.value == 0.0F &&
        mirror_pdf.value.measure == device::ProbabilityMeasure::solid_angle &&
        device::succeeded(mirror.status) && positive(mirror.value) &&
        mirror.probability.measure == device::ProbabilityMeasure::discrete &&
        mirror.probability.value == 1.0F) {
        result.passed_mask |= 1U << 6U;
    }
    result.representative_values[6U] = mirror.value.values[0U];

    const auto transmission_eval = device::specular_transmission_eval(
        coefficient, 1.0F, 1.5F, normal, below, device::TransportMode::radiance);
    const auto transmission_pdf = device::specular_transmission_pdf(
        coefficient, 1.0F, 1.5F, normal, below, device::TransportMode::radiance);
    const auto transmission = device::sample_specular_transmission(coefficient, 1.0F, 1.5F, normal,
                                                                   device::TransportMode::radiance);
    if (device::succeeded(transmission_eval.status) && device::succeeded(transmission_pdf.status) &&
        device::spectrum_is_zero(transmission_eval.value) && transmission_pdf.value.value == 0.0F &&
        transmission_pdf.value.measure == device::ProbabilityMeasure::solid_angle &&
        device::succeeded(transmission.status) && positive(transmission.value) &&
        transmission.probability.measure == device::ProbabilityMeasure::discrete &&
        transmission.incoming_local.z < 0.0F && transmission.eta_scale_multiplier > 1.0F) {
        result.passed_mask |= 1U << 7U;
    }
    result.representative_values[7U] = transmission.eta_scale_multiplier;

    auto mixture = device::ClosureMixtureRecord{};
    mixture.active_count = 2U;
    mixture.closures[0U].kind = device::ClosureKind::lambertian_reflection;
    mixture.closures[0U].lobes =
        device::ScatteringLobe::diffuse | device::ScatteringLobe::reflection;
    mixture.closures[1U].kind = device::ClosureKind::rough_diffuse_reflection;
    mixture.closures[1U].lobes =
        device::ScatteringLobe::diffuse | device::ScatteringLobe::reflection;
    for (auto lane = std::uint32_t{}; lane < device::SpectrumLaneCount; ++lane) {
        mixture.closures[0U].weight[lane] = coefficient.values[lane];
        mixture.closures[1U].weight[lane] = coefficient.values[lane];
    }
    mixture.closures[1U].parameters[0U] = 0.6F;
    mixture.probabilities[0U] = 0.5F;
    mixture.probabilities[1U] = 0.5F;
    mixture.cdf[1U] = 0.5F;
    mixture.cdf[2U] = 1.0F;
    const auto mixture_eval =
        device::closure_mixture_eval(mixture, normal, normal, device::TransportMode::radiance);
    const auto mixture_pdf =
        device::closure_mixture_pdf(mixture, normal, normal, device::TransportMode::radiance);
    const auto mixture_sample = device::sample_closure_mixture(mixture, normal, 0.75F, 0.25F, 0.75F,
                                                               device::TransportMode::radiance);
    if (device::succeeded(mixture_eval.status) && device::succeeded(mixture_pdf.status) &&
        device::succeeded(mixture_sample.status) && positive(mixture_eval.value) &&
        mixture_pdf.value.value > 0.0F && mixture_sample.selected_closure == 1U &&
        mixture_sample.selection_probability.value == 0.5F &&
        device::valid_normalized_mixture_probability(1.0F) &&
        !device::valid_normalized_mixture_probability(0.0F) &&
        !device::valid_normalized_mixture_probability(1.00000012F)) {
        result.passed_mask |= 1U << 8U;
    }
    result.representative_values[8U] = mixture_pdf.value.value;

    auto mixed_measure = device::ClosureMixtureRecord{};
    mixed_measure.active_count = 4U;
    mixed_measure.closures[0U].kind = device::ClosureKind::lambertian_reflection;
    mixed_measure.closures[0U].lobes =
        device::ScatteringLobe::diffuse | device::ScatteringLobe::reflection;
    mixed_measure.closures[1U].kind = device::ClosureKind::specular_reflection;
    mixed_measure.closures[1U].lobes =
        device::ScatteringLobe::specular | device::ScatteringLobe::reflection;
    for (auto index = std::uint32_t{2U}; index < 4U; ++index) {
        mixed_measure.closures[index].kind = device::ClosureKind::specular_transmission;
        mixed_measure.closures[index].lobes =
            device::ScatteringLobe::specular | device::ScatteringLobe::transmission;
        mixed_measure.closures[index].parameters[0U] = 1.0F;
        mixed_measure.closures[index].parameters[1U] = 1.5F;
    }
    for (auto index = std::uint32_t{}; index < 4U; ++index) {
        for (auto lane = std::uint32_t{}; lane < device::SpectrumLaneCount; ++lane) {
            mixed_measure.closures[index].weight[lane] = coefficient.values[lane];
        }
        mixed_measure.probabilities[index] = 0.25F;
        mixed_measure.cdf[index + 1U] = 0.25F * static_cast<float>(index + 1U);
    }
    const auto mixed_measure_eval = device::closure_mixture_eval(mixed_measure, normal, normal,
                                                                 device::TransportMode::radiance);
    const auto mixed_measure_pdf =
        device::closure_mixture_pdf(mixed_measure, normal, normal, device::TransportMode::radiance);
    const auto mixed_mirror = device::sample_closure_mixture(mixed_measure, normal, 0.3F, 0.0F,
                                                             0.0F, device::TransportMode::radiance);
    const auto mixed_transmission = device::sample_closure_mixture(
        mixed_measure, normal, 0.6F, 0.0F, 0.0F, device::TransportMode::radiance);
    if (device::succeeded(mixed_measure_eval.status) &&
        device::succeeded(mixed_measure_pdf.status) && device::succeeded(mixed_mirror.status) &&
        device::succeeded(mixed_transmission.status) && positive(mixed_measure_eval.value) &&
        mixed_measure_pdf.value.measure == device::ProbabilityMeasure::solid_angle &&
        mixed_measure_pdf.value.value > 0.0F &&
        mixed_mirror.probability.measure == device::ProbabilityMeasure::discrete &&
        mixed_mirror.probability.value == 0.25F && mixed_mirror.selected_closure == 1U &&
        mixed_transmission.probability.measure == device::ProbabilityMeasure::discrete &&
        mixed_transmission.probability.value == 0.5F && mixed_transmission.selected_closure == 2U &&
        mixed_transmission.incoming_local.z < 0.0F && positive(mixed_transmission.value)) {
        result.passed_mask |= 1U << 9U;
    }
    result.representative_values[9U] = mixed_transmission.probability.value;
    *output = result;
}

} // namespace

extern "C" int blackframe_cuda_run_transport_lobe_probe(TransportLobeProbeResult* const output,
                                                        int* const device_count) noexcept {
    if (output == nullptr || device_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *output = TransportLobeProbeResult{};
    *device_count = 0;
    auto status = cudaGetDeviceCount(device_count);
    if (status != cudaSuccess || *device_count == 0) {
        return static_cast<int>(status == cudaSuccess ? cudaErrorNoDevice : status);
    }
    TransportLobeProbeResult* device_output = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(TransportLobeProbeResult));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    transport_lobe_probe_kernel<<<1, 1>>>(device_output);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device_output, sizeof(TransportLobeProbeResult),
                            cudaMemcpyDeviceToHost);
    }
    const auto free_status = cudaFree(device_output);
    return static_cast<int>(status == cudaSuccess ? free_status : status);
}
