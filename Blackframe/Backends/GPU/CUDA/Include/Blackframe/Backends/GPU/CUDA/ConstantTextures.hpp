#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <span>
#include <vector>

namespace blackframe::engine {

struct CudaConstantTextureEvaluationOptions final {
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

// These typed entry points evaluate the serialized scene registry on the active CUDA device.
// A missing identifier or a value-kind mismatch fails the complete batch; values are never
// synthesized, converted between color and spectrum, or evaluated on the host as a fallback.
[[nodiscard]] core::Result<std::vector<renderer::TransportScalar>>
evaluate_cuda_constant_float_textures(const CudaSceneSoA& scene,
                                      std::span<const renderer::TextureId> texture_ids,
                                      CudaConstantTextureEvaluationOptions options = {});

[[nodiscard]] core::Result<std::vector<renderer::LinearRGB>>
evaluate_cuda_constant_color_textures(const CudaSceneSoA& scene,
                                      std::span<const renderer::TextureId> texture_ids,
                                      CudaConstantTextureEvaluationOptions options = {});

[[nodiscard]] core::Result<std::vector<renderer::TransportSpectrum>>
evaluate_cuda_constant_spectrum_textures(const CudaSceneSoA& scene,
                                         std::span<const renderer::TextureId> texture_ids,
                                         CudaConstantTextureEvaluationOptions options = {});

} // namespace blackframe::engine
