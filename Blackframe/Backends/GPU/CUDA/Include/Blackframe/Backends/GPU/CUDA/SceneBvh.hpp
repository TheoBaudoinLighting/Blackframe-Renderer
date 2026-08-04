#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Core/Status.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/Shared/SceneBvhAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::engine {

struct CudaSceneBvhBuildOptions final {
    std::uint16_t abi_major{xpu::shared::SceneBvhAbiMajor};
    std::uint16_t abi_minor{xpu::shared::SceneBvhAbiMinor};
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

// Owns the immutable device-resident BLAS/TLAS built for one serialized CUDA
// scene. Construction is deterministic and backend-local; it neither creates
// an Embree scene nor substitutes any CPU traversal implementation.
class CudaSceneBvh final {
  public:
    CudaSceneBvh(const CudaSceneBvh&) = delete;
    CudaSceneBvh& operator=(const CudaSceneBvh&) = delete;
    CudaSceneBvh(CudaSceneBvh&& other) noexcept;
    CudaSceneBvh& operator=(CudaSceneBvh&& other) = delete;
    ~CudaSceneBvh() noexcept = default;

    [[nodiscard]] static core::Result<CudaSceneBvh> build(const CudaSceneSoA& scene,
                                                          CudaSceneBvhBuildOptions options = {});

    [[nodiscard]] const xpu::shared::SceneBvhHeader& header() const noexcept {
        return header_;
    }
    [[nodiscard]] const std::uint8_t* device_data() const noexcept {
        return device_bytes_.data();
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return device_bytes_.size_bytes();
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_bytes_.device_ordinal();
    }
    [[nodiscard]] bool empty() const noexcept {
        return device_bytes_.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(device_bytes_);
    }

    [[nodiscard]] core::Status close();

  private:
    CudaSceneBvh(xpu::shared::SceneBvhHeader header,
                 xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes) noexcept;

    xpu::shared::SceneBvhHeader header_{};
    xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes_{};
};

static_assert(std::is_move_constructible_v<CudaSceneBvh>);
static_assert(!std::is_move_assignable_v<CudaSceneBvh>);
static_assert(!std::is_copy_constructible_v<CudaSceneBvh>);
static_assert(!std::is_copy_assignable_v<CudaSceneBvh>);
static_assert(std::is_nothrow_destructible_v<CudaSceneBvh>);
static_assert(std::is_standard_layout_v<CudaSceneBvhBuildOptions>);

} // namespace blackframe::engine
