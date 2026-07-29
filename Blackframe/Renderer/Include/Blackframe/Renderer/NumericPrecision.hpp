#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {

// Transport storage and arithmetic use the same scalar type on every backend.
using TransportScalar = float;

// The readable scalar reference and its calculations use the wider host type.
using ReferenceScalar = double;

enum class AccumulationPrecision : std::uint8_t {
    float32,
    float64,
};

template <AccumulationPrecision Precision> struct AccumulationScalarTraits;

template <> struct AccumulationScalarTraits<AccumulationPrecision::float32> final {
    using type = TransportScalar;
};

template <> struct AccumulationScalarTraits<AccumulationPrecision::float64> final {
    using type = ReferenceScalar;
};

template <AccumulationPrecision Precision>
using AccumulationScalar = typename AccumulationScalarTraits<Precision>::type;

static_assert(std::is_same_v<TransportScalar, float>);
static_assert(std::is_same_v<ReferenceScalar, double>);
static_assert(sizeof(TransportScalar) == 4);
static_assert(sizeof(ReferenceScalar) == 8);
static_assert(std::numeric_limits<TransportScalar>::is_iec559);
static_assert(std::numeric_limits<ReferenceScalar>::is_iec559);
static_assert(std::numeric_limits<TransportScalar>::digits == 24);
static_assert(std::numeric_limits<ReferenceScalar>::digits == 53);
static_assert(sizeof(AccumulationScalar<AccumulationPrecision::float32>) == 4);
static_assert(sizeof(AccumulationScalar<AccumulationPrecision::float64>) == 8);

} // namespace blackframe::renderer
