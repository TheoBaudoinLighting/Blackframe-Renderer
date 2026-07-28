#pragma once

#include <cstdint>
#include <string>

namespace blackframe::xpu {

enum class DeviceKind : std::uint8_t {
    Unknown,
    Cpu,
    Gpu,
    Accelerator,
};

struct DeviceIdentifier final {
    std::uint64_t high{};
    std::uint64_t low{};

    [[nodiscard]] friend constexpr auto operator==(const DeviceIdentifier&,
                                                   const DeviceIdentifier&) noexcept
        -> bool = default;
};

struct DeviceDescriptor final {
    DeviceIdentifier identifier;
    DeviceKind kind{DeviceKind::Unknown};
    std::string backend_name;
    std::string name;
    std::string vendor;
};

} // namespace blackframe::xpu
