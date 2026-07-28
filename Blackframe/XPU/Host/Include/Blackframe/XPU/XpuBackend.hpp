#pragma once

#include <Blackframe/Extensions/ExtensionModule.hpp>
#include <Blackframe/XPU/DeviceDescriptor.hpp>
#include <Blackframe/XPU/XpuBackendAbi.h>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace blackframe::xpu {

enum class XpuHostErrorCode : std::uint8_t {
    ExtensionLoadFailed,
    InterfaceUnavailable,
    MalformedBackendApi,
    EnumerationFailed,
    InvalidDeviceCount,
    MalformedDeviceDescriptor,
};

struct XpuHostError final {
    XpuHostErrorCode code{};
    blackframe_status extension_status{BLACKFRAME_STATUS_SUCCESS};
    std::filesystem::path extension_path;
    std::string message;
};

class XpuBackend final {
  public:
    ~XpuBackend() = default;

    XpuBackend(const XpuBackend&) = delete;
    auto operator=(const XpuBackend&) -> XpuBackend& = delete;

    XpuBackend(XpuBackend&&) noexcept = default;
    auto operator=(XpuBackend&&) noexcept -> XpuBackend& = default;

    [[nodiscard]] static auto load(const std::filesystem::path& absolute_extension_path)
        -> std::expected<XpuBackend, XpuHostError>;

    [[nodiscard]] auto enumerate_devices() const
        -> std::expected<std::vector<DeviceDescriptor>, XpuHostError>;

    [[nodiscard]] auto name() const noexcept -> std::string_view;
    [[nodiscard]] auto extension_path() const noexcept -> const std::filesystem::path&;

  private:
    XpuBackend(extensions::ExtensionModule extension, blackframe_xpu_backend_api_v1 backend_api,
               std::string backend_name) noexcept;

    extensions::ExtensionModule extension_;
    blackframe_xpu_backend_api_v1 backend_api_{};
    std::string backend_name_;
};

} // namespace blackframe::xpu
