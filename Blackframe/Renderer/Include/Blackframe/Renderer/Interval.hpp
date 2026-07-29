#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <cmath>
#include <limits>

namespace blackframe::renderer {

template <GeometryScalar Scalar> class IntervalT final {
  public:
    [[nodiscard]] static core::Result<IntervalT> closed(const Scalar lower, const Scalar upper) {
        if (std::isnan(lower) || std::isnan(upper) || lower > upper) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "An interval requires ordered endpoints without NaN.",
            });
        }
        return IntervalT{lower, upper, false};
    }

    [[nodiscard]] static constexpr IntervalT empty() noexcept {
        return IntervalT{Scalar{0}, Scalar{0}, true};
    }

    [[nodiscard]] static constexpr IntervalT unbounded() noexcept {
        return IntervalT{-std::numeric_limits<Scalar>::infinity(),
                         std::numeric_limits<Scalar>::infinity(), false};
    }

    [[nodiscard]] static constexpr IntervalT nonnegative() noexcept {
        return IntervalT{Scalar{0}, std::numeric_limits<Scalar>::infinity(), false};
    }

    [[nodiscard]] constexpr bool is_empty() const noexcept {
        return empty_;
    }
    [[nodiscard]] constexpr Scalar lower() const noexcept {
        return lower_;
    }
    [[nodiscard]] constexpr Scalar upper() const noexcept {
        return upper_;
    }

    [[nodiscard]] constexpr bool contains(const Scalar value) const noexcept {
        return !empty_ && !std::isnan(value) && value >= lower_ && value <= upper_;
    }

    [[nodiscard]] constexpr bool overlaps(const IntervalT& other) const noexcept {
        return !empty_ && !other.empty_ && lower_ <= other.upper_ && other.lower_ <= upper_;
    }

    [[nodiscard]] constexpr IntervalT intersection_with(const IntervalT& other) const noexcept {
        if (!overlaps(other)) {
            return empty();
        }
        return IntervalT{lower_ > other.lower_ ? lower_ : other.lower_,
                         upper_ < other.upper_ ? upper_ : other.upper_, false};
    }

  private:
    constexpr IntervalT(const Scalar lower, const Scalar upper, const bool empty) noexcept
        : lower_{lower}, upper_{upper}, empty_{empty} {}

    Scalar lower_;
    Scalar upper_;
    bool empty_;
};

using Interval = IntervalT<TransportScalar>;
using ReferenceInterval = IntervalT<ReferenceScalar>;

} // namespace blackframe::renderer
