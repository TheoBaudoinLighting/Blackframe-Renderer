#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace blackframe::renderer {

inline constexpr std::uint32_t CapabilityManifestSchemaVersion = 1;

enum class BackendCapabilityStatus : std::uint8_t {
    supported,
    experimental,
    unavailable,
};

struct BackendCapability {
    std::string_view identifier;
    BackendCapabilityStatus status;
};

[[nodiscard]] std::string_view
backend_capability_status_name(BackendCapabilityStatus status) noexcept;
[[nodiscard]] std::span<const BackendCapability> backend_capabilities() noexcept;
[[nodiscard]] std::string backend_capability_manifest();

} // namespace blackframe::renderer
