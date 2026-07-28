#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/IPC/LocalTransport.hpp>
#include <Blackframe/XPU/DeviceDescriptor.hpp>
#include <Blackframe/XPU/XpuBackend.hpp>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace blackframe::engine {

// Owns accepted XPU modules for the lifetime of the process and adapts the
// small IPC control protocol to engine state. Rendering is deliberately absent
// from the foundation milestone.
class Engine final {
  public:
    [[nodiscard]] core::Status load_xpu_extension(const std::filesystem::path& absolute_path);

    [[nodiscard]] core::Result<std::vector<xpu::DeviceDescriptor>> enumerate_devices() const;

    [[nodiscard]] std::expected<ipc::ServerResponse, ipc::RequestHandlingError>
    handle_request(const ipc::ProtocolMessage& request) const;

    [[nodiscard]] std::size_t xpu_backend_count() const noexcept;

  private:
    std::vector<xpu::XpuBackend> xpu_backends_;
};

} // namespace blackframe::engine
