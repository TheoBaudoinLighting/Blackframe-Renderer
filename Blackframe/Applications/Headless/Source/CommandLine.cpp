#include "CommandLine.hpp"

#include "DefaultLocalEndpoint.hpp"

#include <string_view>

namespace blackframe::application {
namespace {

[[nodiscard]] core::Error command_line_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Result<ipc::Command> parse_request_command(const std::string_view value) {
    if (value == "ping") {
        return ipc::Command::Ping;
    }
    if (value == "version") {
        return ipc::Command::QueryVersion;
    }
    if (value == "devices") {
        return ipc::Command::EnumerateDevices;
    }
    if (value == "shutdown") {
        return ipc::Command::Shutdown;
    }
    return std::unexpected(command_line_error("Unknown request command: " + std::string{value}));
}

} // namespace

core::Result<CommandLine> parse_command_line(const int argument_count,
                                             const char* const* const arguments) {
    auto command_line = CommandLine{
        .operation = Operation::show_help,
        .request_command = ipc::Command::Ping,
        .endpoint = default_local_endpoint().address,
        .xpu_plugins = {},
    };

    if (argument_count <= 1) {
        return command_line;
    }

    const auto operation = std::string_view{arguments[1]};
    if (operation == "--help" || operation == "-h" || operation == "help") {
        return command_line;
    }
    if (operation == "--version" || operation == "version") {
        command_line.operation = Operation::show_version;
        return command_line;
    }
    if (operation == "serve") {
        command_line.operation = Operation::serve;
    } else if (operation == "request") {
        command_line.operation = Operation::request;
        if (argument_count <= 2) {
            return std::unexpected(command_line_error("The request command is missing."));
        }
        auto request_command = parse_request_command(arguments[2]);
        if (!request_command) {
            return std::unexpected(std::move(request_command.error()));
        }
        command_line.request_command = *request_command;
    } else {
        return std::unexpected(command_line_error("Unknown operation: " + std::string{operation}));
    }

    int argument_index = command_line.operation == Operation::request ? 3 : 2;
    while (argument_index < argument_count) {
        const auto argument = std::string_view{arguments[argument_index]};
        if (argument == "--endpoint") {
            if (++argument_index >= argument_count) {
                return std::unexpected(command_line_error("The --endpoint value is missing."));
            }
            command_line.endpoint = arguments[argument_index];
        } else if (argument == "--xpu-plugin") {
            if (command_line.operation != Operation::serve) {
                return std::unexpected(
                    command_line_error("--xpu-plugin is valid only for the serve operation."));
            }
            if (++argument_index >= argument_count) {
                return std::unexpected(command_line_error("The --xpu-plugin path is missing."));
            }
            command_line.xpu_plugins.emplace_back(arguments[argument_index]);
        } else {
            return std::unexpected(command_line_error("Unknown option: " + std::string{argument}));
        }
        ++argument_index;
    }

    return command_line;
}

std::string command_line_usage() {
    return "Blackframe\n"
           "Usage:\n"
           "  Blackframe serve [--endpoint <address>] [--xpu-plugin <absolute-path>]...\n"
           "  Blackframe request <ping|version|devices|shutdown> [--endpoint <address>]\n"
           "  Blackframe --version\n";
}

} // namespace blackframe::application
