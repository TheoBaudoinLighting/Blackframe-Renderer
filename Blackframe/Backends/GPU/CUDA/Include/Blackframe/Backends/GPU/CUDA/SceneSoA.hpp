#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::engine {

struct CudaSceneSoAUploadOptions final {
    std::uint16_t abi_major{xpu::shared::SceneSoaAbiMajor};
    std::uint16_t abi_minor{xpu::shared::SceneSoaAbiMinor};
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

// Owns one immutable, canonical scene blob in CUDA device memory. The blob has
// no native pointers: every column is addressed by a checked byte offset from
// its versioned header, so allocation identity cannot affect its contents.
// CPU MeshAreaLight models are derived caches rather than a second scene truth;
// the blob carries their stable instance registry plus the exact mesh,
// material emission, and resolved transforms needed to rebuild them.
class CudaSceneSoA final {
  public:
    CudaSceneSoA(const CudaSceneSoA&) = delete;
    CudaSceneSoA& operator=(const CudaSceneSoA&) = delete;
    CudaSceneSoA(CudaSceneSoA&& other) noexcept;
    CudaSceneSoA& operator=(CudaSceneSoA&& other) = delete;
    ~CudaSceneSoA() noexcept = default;

    [[nodiscard]] static core::Result<CudaSceneSoA> upload(const FrameScene& scene,
                                                           CudaSceneSoAUploadOptions options = {});

    [[nodiscard]] const xpu::shared::SceneSoaHeader& header() const noexcept {
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
    CudaSceneSoA(xpu::shared::SceneSoaHeader header,
                 xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes) noexcept;

    xpu::shared::SceneSoaHeader header_{};
    xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes_{};
};

static_assert(std::is_move_constructible_v<CudaSceneSoA>);
static_assert(!std::is_move_assignable_v<CudaSceneSoA>);
static_assert(!std::is_copy_constructible_v<CudaSceneSoA>);
static_assert(!std::is_copy_assignable_v<CudaSceneSoA>);
static_assert(std::is_nothrow_destructible_v<CudaSceneSoA>);
static_assert(std::is_standard_layout_v<CudaSceneSoAUploadOptions>);

} // namespace blackframe::engine
