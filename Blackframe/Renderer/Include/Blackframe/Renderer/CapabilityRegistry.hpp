#pragma once

#include <Blackframe/Core/Status.hpp>
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
    std::string_view required_dependency;
};

[[nodiscard]] std::string_view
backend_capability_status_name(BackendCapabilityStatus status) noexcept;
[[nodiscard]] std::span<const BackendCapability> backend_capabilities() noexcept;
[[nodiscard]] std::string backend_capability_manifest();

// Every rendering entry point must pass this gate before creating a backend,
// allocating backend resources, or dispatching work.
[[nodiscard]] core::Status require_backend_capability(std::string_view identifier);

} // namespace blackframe::renderer
