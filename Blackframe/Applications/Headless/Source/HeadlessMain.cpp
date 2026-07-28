#include "CommandLine.hpp"

#include <Blackframe/Core/Version.hpp>
#include <Blackframe/Engine/Engine.hpp>
#include <Blackframe/IPC/LocalTransport.hpp>
#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

[[nodiscard]] int serve(const blackframe::application::CommandLine& command_line) {
    auto engine = blackframe::engine::Engine{};
    for (const auto& plugin_path : command_line.xpu_plugins) {
        auto loaded = engine.load_xpu_extension(plugin_path);
        if (!loaded) {
            std::cerr << "Blackframe: cannot load XPU extension '" << plugin_path.string()
                      << "': " << loaded.error().message << '\n';
            return 1;
        }
    }

    auto server = blackframe::ipc::LocalIpcServer::Listen(
        blackframe::ipc::LocalEndpoint{.address = command_line.endpoint});
    if (!server) {
        std::cerr << "Blackframe: cannot listen on '" << command_line.endpoint
                  << "': " << server.error().message << '\n';
        return 1;
    }

    std::cout << "Blackframe " << blackframe::core::version_string() << " listening on "
              << command_line.endpoint << '\n';

    auto result = server->Run([&engine](const blackframe::ipc::ProtocolMessage& request) {
        return engine.handle_request(request);
    });
    if (!result) {
        std::cerr << "Blackframe: IPC server failed: " << result.error().message << '\n';
        return 1;
    }

    return 0;
}

[[nodiscard]] int request(const blackframe::application::CommandLine& command_line) {
    constexpr auto request_id = std::uint64_t{1};
    const auto request_message =
        blackframe::ipc::MakeProtocolRequest(command_line.request_command, request_id);
    auto response = blackframe::ipc::LocalIpcClient::Exchange(
        blackframe::ipc::LocalEndpoint{.address = command_line.endpoint}, request_message);
    if (!response) {
        std::cerr << "Blackframe: IPC request failed: " << response.error().message << '\n';
        return 1;
    }
    if (response->status != blackframe::ipc::Status::Ok) {
        std::cerr << "Blackframe: the engine rejected the request with status "
                  << static_cast<unsigned>(response->status) << '\n';
        return 1;
    }

    switch (command_line.request_command) {
    case blackframe::ipc::Command::Ping:
        std::cout << "pong\n";
        break;
    case blackframe::ipc::Command::QueryVersion: {
        auto version = blackframe::ipc::DecodeVersionInformation(response->payload);
        if (!version) {
            std::cerr << "Blackframe: invalid version response: " << version.error().message
                      << '\n';
            return 1;
        }
        std::cout << "Blackframe " << blackframe::core::version_string() << " (IPC "
                  << version->protocol_version << ")\n";
        break;
    }
    case blackframe::ipc::Command::EnumerateDevices: {
        auto devices = blackframe::ipc::DecodeDeviceDescriptions(response->payload);
        if (!devices) {
            std::cerr << "Blackframe: invalid device response: " << devices.error().message << '\n';
            return 1;
        }
        for (const auto& device : *devices) {
            std::cout << device.backend_name << ": " << device.device_name << '\n';
        }
        break;
    }
    case blackframe::ipc::Command::Shutdown:
        std::cout << "shutdown accepted\n";
        break;
    }
    return 0;
}

} // namespace

int main(const int argument_count, const char* const* const arguments) {
    const auto command_line =
        blackframe::application::parse_command_line(argument_count, arguments);
    if (!command_line) {
        std::cerr << "Blackframe: " << command_line.error().message << '\n'
                  << blackframe::application::command_line_usage();
        return 2;
    }

    switch (command_line->operation) {
    case blackframe::application::Operation::show_help:
        std::cout << blackframe::application::command_line_usage();
        return 0;
    case blackframe::application::Operation::show_version:
        std::cout << blackframe::core::product_name() << ' ' << blackframe::core::version_string()
                  << '\n';
        return 0;
    case blackframe::application::Operation::show_capabilities:
        std::cout << blackframe::renderer::backend_capability_manifest();
        return 0;
    case blackframe::application::Operation::serve:
        return serve(*command_line);
    case blackframe::application::Operation::request:
        return request(*command_line);
    }
    return 2;
}
