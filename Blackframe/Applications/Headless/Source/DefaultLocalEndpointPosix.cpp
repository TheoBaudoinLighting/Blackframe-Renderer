#include "DefaultLocalEndpoint.hpp"

#include <filesystem>
#include <string>
#include <unistd.h>

namespace blackframe::application {

ipc::LocalEndpoint default_local_endpoint() {
    const auto file_name = std::string{"blackframe-"} +
                           std::to_string(static_cast<unsigned long long>(::getuid())) + ".sock";
    return ipc::LocalEndpoint{
        .address = (std::filesystem::temp_directory_path() / file_name).string(),
    };
}

} // namespace blackframe::application
