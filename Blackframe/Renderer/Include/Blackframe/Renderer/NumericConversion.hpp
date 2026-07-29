#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>

namespace blackframe::renderer {

// Every finite float has an exact representation as a double.
[[nodiscard]] constexpr ReferenceScalar to_reference_scalar(const TransportScalar value) noexcept {
    return static_cast<ReferenceScalar>(value);
}

// Narrowing is deliberately checked. Rounding to the nearest representable transport value is
// allowed, while non-finite values, overflow, and underflow to zero are reported to the caller.
[[nodiscard]] core::Result<TransportScalar> to_transport_scalar(ReferenceScalar value);

} // namespace blackframe::renderer
