#include <Blackframe/Core/Version.hpp>
#include <format>

namespace blackframe::core {

Version current_version() noexcept {
    return {
        .major = BLACKFRAME_VERSION_MAJOR,
        .minor = BLACKFRAME_VERSION_MINOR,
        .patch = BLACKFRAME_VERSION_PATCH,
    };
}

std::string version_string() {
    const auto version = current_version();
    return std::format("{}.{}.{}", version.major, version.minor, version.patch);
}

} // namespace blackframe::core
