#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace blackframe::core {

enum class StatusCode : std::uint32_t {
    success = 0,
    invalid_argument,
    unavailable,
    not_found,
    incompatible,
    resource_exhausted,
    protocol_error,
    platform_error,
    internal_error,
};

struct Error {
    StatusCode code{StatusCode::internal_error};
    std::string message;
};

template <typename Value> using Result = std::expected<Value, Error>;

using Status = std::expected<void, Error>;

[[nodiscard]] std::string_view status_code_name(StatusCode code) noexcept;

} // namespace blackframe::core
