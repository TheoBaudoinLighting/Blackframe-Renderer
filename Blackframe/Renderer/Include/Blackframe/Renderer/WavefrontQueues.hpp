#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

enum class WavefrontQueueKind : std::uint8_t {
    camera = 0U,
    ray = 1U,
    hit = 2U,
    miss = 3U,
    shade = 4U,
    shadow = 5U,
    continuation = 6U,
};

[[nodiscard]] constexpr bool is_known_wavefront_queue_kind(const WavefrontQueueKind kind) noexcept {
    switch (kind) {
    case WavefrontQueueKind::camera:
    case WavefrontQueueKind::ray:
    case WavefrontQueueKind::hit:
    case WavefrontQueueKind::miss:
    case WavefrontQueueKind::shade:
    case WavefrontQueueKind::shadow:
    case WavefrontQueueKind::continuation:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr std::string_view
wavefront_queue_kind_name(const WavefrontQueueKind kind) noexcept {
    switch (kind) {
    case WavefrontQueueKind::camera:
        return "camera";
    case WavefrontQueueKind::ray:
        return "ray";
    case WavefrontQueueKind::hit:
        return "hit";
    case WavefrontQueueKind::miss:
        return "miss";
    case WavefrontQueueKind::shade:
        return "shade";
    case WavefrontQueueKind::shadow:
        return "shadow";
    case WavefrontQueueKind::continuation:
        return "continuation";
    }
    return "unknown";
}

// Queue entries only identify path slots owned by the surrounding wavefront streams. Every 32-bit
// value is preserved exactly; callers must validate it against the owning stream before enqueueing.
// Stage-specific ray, hit, surface, and shadow data remain in their owning streams instead of being
// duplicated in queue records. This host execution record is not a wire format or a host/device
// ABI.
struct WavefrontPathSlot final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const WavefrontPathSlot&) const noexcept = default;
};

// Capacity exhaustion is a trivial status so the overflow path never allocates an error message.
// A rejected push never grows, drops, overwrites, or partially appends queue entries.
enum class WavefrontQueuePushStatus : std::uint8_t {
    pushed = 0U,
    capacity_exhausted = 1U,
};

namespace wavefront_queue_detail {

template <WavefrontQueueKind Kind> class BoundedWavefrontQueueT final {
    static_assert(is_known_wavefront_queue_kind(Kind));

  public:
    BoundedWavefrontQueueT(const BoundedWavefrontQueueT&) = delete;
    BoundedWavefrontQueueT(BoundedWavefrontQueueT&& other) noexcept;
    BoundedWavefrontQueueT& operator=(const BoundedWavefrontQueueT&) = delete;
    BoundedWavefrontQueueT& operator=(BoundedWavefrontQueueT&&) = delete;
    ~BoundedWavefrontQueueT() = default;

    [[nodiscard]] static core::Result<BoundedWavefrontQueueT> create(std::size_t capacity);

    [[nodiscard]] static constexpr WavefrontQueueKind kind() noexcept {
        return Kind;
    }

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t remaining_capacity() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool full() const noexcept;

    // The backing address remains stable from creation until move or destruction. A view's active
    // range expires when the queue is cleared, even though a newly acquired view still uses the
    // same preallocated storage. Rvalue access is forbidden so a temporary queue cannot produce a
    // dangling view.
    [[nodiscard]] std::span<const WavefrontPathSlot> entries() const& noexcept;
    [[nodiscard]] std::span<const WavefrontPathSlot> entries() && = delete;
    [[nodiscard]] std::span<const WavefrontPathSlot> entries() const&& = delete;

    [[nodiscard]] WavefrontQueuePushStatus push(WavefrontPathSlot slot) noexcept;
    [[nodiscard]] WavefrontQueuePushStatus
    push_batch(std::span<const WavefrontPathSlot> slots) noexcept;
    void clear() noexcept;

  private:
    BoundedWavefrontQueueT() = default;
    void swap(BoundedWavefrontQueueT& other) noexcept;

    std::vector<WavefrontPathSlot> storage_;
    std::size_t active_size_{};
};

extern template class BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;

} // namespace wavefront_queue_detail

using CameraQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
using RayQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
using HitQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
using MissQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
using ShadeQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
using ShadowQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
using ContinuationQueue =
    wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;

static_assert(std::is_standard_layout_v<WavefrontPathSlot>);
static_assert(std::is_trivially_copyable_v<WavefrontPathSlot>);
static_assert(sizeof(WavefrontPathSlot) == sizeof(std::uint32_t));
static_assert(sizeof(WavefrontQueueKind) == sizeof(std::uint8_t));
static_assert(sizeof(WavefrontQueuePushStatus) == sizeof(std::uint8_t));
static_assert(!std::is_default_constructible_v<CameraQueue>);
static_assert(!std::is_copy_constructible_v<CameraQueue>);
static_assert(std::is_nothrow_move_constructible_v<CameraQueue>);

} // namespace blackframe::renderer
