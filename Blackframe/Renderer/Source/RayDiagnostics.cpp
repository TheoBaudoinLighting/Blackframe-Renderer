#include <Blackframe/Renderer/RayDiagnostics.hpp>
#include <bit>
#include <cstdint>
#include <string>
#include <type_traits>

namespace blackframe::renderer {
namespace {

static_assert(CurrentRayDiagnosticSchemaVersion == 1);

template <typename Unsigned> void append_hex(std::string& output, const Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    constexpr auto digits = "0123456789abcdef";
    output += "0x";
    for (auto index = sizeof(Unsigned) * 2; index > 0; --index) {
        const auto shift = (index - 1) * 4;
        const auto nibble = static_cast<std::size_t>((value >> shift) & Unsigned{0xFU});
        output += digits[nibble];
    }
}

template <GeometryScalar Scalar> void append_scalar(std::string& output, const Scalar value) {
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        append_hex(output, std::bit_cast<std::uint32_t>(value));
    } else {
        append_hex(output, std::bit_cast<std::uint64_t>(value));
    }
}

template <GeometryScalar Scalar> [[nodiscard]] std::string serialize(const RayT<Scalar>& ray) {
    auto output = std::string{};
    output.reserve(std::same_as<Scalar, TransportScalar> ? 320U : 480U);
    output += R"({"schema_version":1,"precision":")";
    output += std::same_as<Scalar, TransportScalar> ? "float32" : "float64";
    output += R"(","origin_bits":[")";
    append_scalar(output, ray.origin().x);
    output += R"(",")";
    append_scalar(output, ray.origin().y);
    output += R"(",")";
    append_scalar(output, ray.origin().z);
    output += R"("],"direction_bits":[")";
    append_scalar(output, ray.direction().x);
    output += R"(",")";
    append_scalar(output, ray.direction().y);
    output += R"(",")";
    append_scalar(output, ray.direction().z);
    output += R"("],"t_min_bits":")";
    append_scalar(output, ray.t_min());
    output += R"(","t_max_bits":")";
    append_scalar(output, ray.t_max());
    output += R"(","time_bits":")";
    append_scalar(output, ray.time());
    output += R"(","mask":")";
    append_hex(output, ray.mask());
    output += R"(","current_medium":")";
    append_hex(output, ray.current_medium().value);
    output += R"("})";
    return output;
}

} // namespace

std::string serialize_ray_diagnostic(const Ray& ray) {
    return serialize(ray);
}

std::string serialize_ray_diagnostic(const ReferenceRay& ray) {
    return serialize(ray);
}

} // namespace blackframe::renderer
