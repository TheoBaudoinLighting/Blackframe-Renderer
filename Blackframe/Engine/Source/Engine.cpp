#include <Blackframe/Engine/Engine.hpp>
#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error make_xpu_error(const xpu::XpuHostError& error) {
    return core::Error{
        .code = core::StatusCode::unavailable,
        .message = error.message,
    };
}

void append_identifier_half(std::array<std::byte, 16>& output, const std::size_t offset,
                            const std::uint64_t value) noexcept {
    for (std::size_t byte_index = 0; byte_index < sizeof(value); ++byte_index) {
        const auto shift = static_cast<unsigned>(byte_index * 8U);
        output[offset + byte_index] = static_cast<std::byte>((value >> shift) & 0xFFU);
    }
}

[[nodiscard]] ipc::DeviceKind to_ipc_device_kind(const xpu::DeviceKind kind) noexcept {
    switch (kind) {
    case xpu::DeviceKind::Cpu:
        return ipc::DeviceKind::Cpu;
    case xpu::DeviceKind::Gpu:
        return ipc::DeviceKind::Gpu;
    case xpu::DeviceKind::Accelerator:
        return ipc::DeviceKind::Accelerator;
    case xpu::DeviceKind::Unknown:
        return ipc::DeviceKind::Unknown;
    }
    return ipc::DeviceKind::Unknown;
}

[[nodiscard]] ipc::RequestHandlingError make_request_error(const ipc::Status status,
                                                           std::string message) {
    return ipc::RequestHandlingError{
        .status = status,
        .message = std::move(message),
    };
}

} // namespace

core::Status Engine::load_xpu_extension(const std::filesystem::path& absolute_path) {
    auto backend = xpu::XpuBackend::load(absolute_path);
    if (!backend) {
        return std::unexpected(make_xpu_error(backend.error()));
    }

    for (const auto& loaded_backend : xpu_backends_) {
        if (loaded_backend.extension_path() == backend->extension_path()) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The XPU extension is already loaded.",
            });
        }
    }

    xpu_backends_.push_back(std::move(*backend));
    return {};
}

core::Result<std::vector<xpu::DeviceDescriptor>> Engine::enumerate_devices() const {
    std::vector<xpu::DeviceDescriptor> devices;

    for (const auto& backend : xpu_backends_) {
        auto backend_devices = backend.enumerate_devices();
        if (!backend_devices) {
            return std::unexpected(make_xpu_error(backend_devices.error()));
        }

        devices.reserve(devices.size() + backend_devices->size());
        for (auto& device : *backend_devices) {
            devices.push_back(std::move(device));
        }
    }

    return devices;
}

std::expected<ipc::ServerResponse, ipc::RequestHandlingError>
Engine::handle_request(const ipc::ProtocolMessage& request) const {
    switch (request.command) {
    case ipc::Command::Ping:
        return ipc::ServerResponse{};

    case ipc::Command::QueryVersion: {
        auto payload = ipc::EncodeVersionInformation(ipc::VersionInformation{});
        if (!payload) {
            return std::unexpected(
                make_request_error(ipc::Status::InternalError, payload.error().message));
        }
        return ipc::ServerResponse{
            .status = ipc::Status::Ok,
            .payload = std::move(*payload),
            .stop_server_after_response = false,
        };
    }

    case ipc::Command::EnumerateDevices: {
        auto devices = enumerate_devices();
        if (!devices) {
            return std::unexpected(
                make_request_error(ipc::Status::Unavailable, devices.error().message));
        }

        std::vector<ipc::DeviceDescription> descriptions;
        descriptions.reserve(devices->size());
        for (const auto& device : *devices) {
            auto identifier = std::array<std::byte, 16>{};
            append_identifier_half(identifier, 0, device.identifier.high);
            append_identifier_half(identifier, 8, device.identifier.low);
            descriptions.push_back(ipc::DeviceDescription{
                .identifier = identifier,
                .kind = to_ipc_device_kind(device.kind),
                .backend_name = device.backend_name,
                .device_name = device.name,
            });
        }

        auto payload = ipc::EncodeDeviceDescriptions(descriptions);
        if (!payload) {
            return std::unexpected(
                make_request_error(ipc::Status::InternalError, payload.error().message));
        }
        return ipc::ServerResponse{
            .status = ipc::Status::Ok,
            .payload = std::move(*payload),
            .stop_server_after_response = false,
        };
    }

    case ipc::Command::Shutdown:
        return ipc::ServerResponse{
            .status = ipc::Status::Ok,
            .payload = {},
            .stop_server_after_response = true,
        };
    }

    return std::unexpected(
        make_request_error(ipc::Status::UnsupportedCommand, "The command is not supported."));
}

std::size_t Engine::xpu_backend_count() const noexcept {
    return xpu_backends_.size();
}

} // namespace blackframe::engine
