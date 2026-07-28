#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/IPC/Protocol.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace blackframe::application {

enum class Operation {
    show_help,
    show_version,
    show_capabilities,
    serve,
    request,
};

struct CommandLine {
    Operation operation{Operation::show_help};
    ipc::Command request_command{ipc::Command::Ping};
    std::string endpoint;
    std::vector<std::filesystem::path> xpu_plugins;
};

[[nodiscard]] core::Result<CommandLine> parse_command_line(int argument_count,
                                                           const char* const* arguments);
[[nodiscard]] std::string command_line_usage();

} // namespace blackframe::application
