#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <cstdint>
#include <string>

namespace blackframe::renderer {

inline constexpr std::uint32_t CurrentPathStateDiagnosticSchemaVersion = 1;

// Floating-point values are emitted as fixed-width IEEE bit patterns. The requested version is
// mandatory so an unsupported diagnostic schema cannot silently fall back to the current one.
[[nodiscard]] core::Result<std::string>
serialize_path_state_diagnostic(const PathState& state, std::uint32_t schema_version);
[[nodiscard]] core::Result<std::string>
serialize_path_state_diagnostic(const ReferencePathState& state, std::uint32_t schema_version);

} // namespace blackframe::renderer
