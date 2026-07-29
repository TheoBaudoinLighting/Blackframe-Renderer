#include <Blackframe/Renderer/NumericConversion.hpp>
#include <cmath>
#include <limits>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Result<TransportScalar> conversion_error(const char* message) {
    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    });
}

} // namespace

core::Result<TransportScalar> to_transport_scalar(const ReferenceScalar value) {
    if (!std::isfinite(value)) {
        return conversion_error("A reference scalar must be finite before transport conversion.");
    }

    constexpr auto transport_max =
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::max());
    constexpr auto transport_lowest =
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::lowest());
    if (value < transport_lowest || value > transport_max) {
        return conversion_error("A reference scalar exceeds the finite transport range.");
    }

    const auto converted = static_cast<TransportScalar>(value);
    if (value != ReferenceScalar{0} && converted == TransportScalar{0}) {
        return conversion_error("A reference scalar underflows to zero in transport precision.");
    }

    return converted;
}

} // namespace blackframe::renderer
