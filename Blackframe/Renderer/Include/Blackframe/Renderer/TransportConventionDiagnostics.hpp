#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstdint>
#include <string>

namespace blackframe::renderer {

inline constexpr std::uint32_t CurrentBsdfConventionSchemaVersion = 1;

// Returns the canonical machine-readable BSDF convention document. The requested schema is
// mandatory; an unsupported version is an error rather than an alias for the current contract.
[[nodiscard]] core::Result<std::string> dump_bsdf_conventions(std::uint32_t schema_version);

} // namespace blackframe::renderer
