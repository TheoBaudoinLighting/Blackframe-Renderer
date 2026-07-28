#include <Blackframe/Core/Status.hpp>

namespace blackframe::core {

std::string_view status_code_name(const StatusCode code) noexcept {
    switch (code) {
    case StatusCode::success:
        return "success";
    case StatusCode::invalid_argument:
        return "invalid_argument";
    case StatusCode::unavailable:
        return "unavailable";
    case StatusCode::not_found:
        return "not_found";
    case StatusCode::incompatible:
        return "incompatible";
    case StatusCode::resource_exhausted:
        return "resource_exhausted";
    case StatusCode::protocol_error:
        return "protocol_error";
    case StatusCode::platform_error:
        return "platform_error";
    case StatusCode::internal_error:
        return "internal_error";
    }

    return "unknown";
}

} // namespace blackframe::core
