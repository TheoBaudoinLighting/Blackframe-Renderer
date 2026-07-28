#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace blackframe::core {

struct Version {
    std::uint32_t major{};
    std::uint32_t minor{};
    std::uint32_t patch{};
};

[[nodiscard]] Version current_version() noexcept;
[[nodiscard]] std::string version_string();
[[nodiscard]] constexpr std::string_view product_name() noexcept {
    return "Blackframe";
}

} // namespace blackframe::core
