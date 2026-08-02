#pragma once

#include <Blackframe/Core/Status.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
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

// Publication and release are explicit state transitions. In particular, an empty published batch
// still remains pending until release, so queue generations can never be silently coalesced.
enum class WavefrontQueuePublishStatus : std::uint8_t {
    published = 0U,
    read_buffer_pending = 1U,
};

enum class WavefrontQueueReleaseStatus : std::uint8_t {
    released = 0U,
    no_read_buffer_pending = 1U,
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

// Owns two equally sized queues whose roles alternate without moving their storage. Producers can
// fill the write buffer while consumers inspect the immutable read buffer. Publishing and releasing
// are externally synchronized barrier operations; the type does not claim concurrent thread safety.
template <WavefrontQueueKind Kind> class DoubleBufferedWavefrontQueueT final {
    static_assert(is_known_wavefront_queue_kind(Kind));

    using Queue = BoundedWavefrontQueueT<Kind>;

  public:
    DoubleBufferedWavefrontQueueT(const DoubleBufferedWavefrontQueueT&) = delete;
    DoubleBufferedWavefrontQueueT(DoubleBufferedWavefrontQueueT&& other) noexcept;
    DoubleBufferedWavefrontQueueT& operator=(const DoubleBufferedWavefrontQueueT&) = delete;
    DoubleBufferedWavefrontQueueT& operator=(DoubleBufferedWavefrontQueueT&&) = delete;
    ~DoubleBufferedWavefrontQueueT() = default;

    // Capacity is per buffer. Creation succeeds only after both fixed-capacity queues exist.
    [[nodiscard]] static core::Result<DoubleBufferedWavefrontQueueT> create(std::size_t capacity);

    [[nodiscard]] static constexpr WavefrontQueueKind kind() noexcept {
        return Kind;
    }

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t write_size() const noexcept;
    [[nodiscard]] std::size_t write_remaining_capacity() const noexcept;
    [[nodiscard]] bool has_pending_read() const noexcept;
    [[nodiscard]] bool write_empty() const noexcept;
    [[nodiscard]] bool write_full() const noexcept;

    // A disengaged optional means no generation is pending; an engaged empty span is an explicitly
    // published empty generation. Views are read-only. Write views must be reacquired after a push,
    // and all views must be reacquired after a publish or release transition. Mutable queue objects
    // are intentionally never exposed, preserving ownership of both allocations.
    [[nodiscard]] std::optional<std::span<const WavefrontPathSlot>>
    pending_read_entries() const& noexcept;
    [[nodiscard]] std::optional<std::span<const WavefrontPathSlot>>
    pending_read_entries() && = delete;
    [[nodiscard]] std::optional<std::span<const WavefrontPathSlot>>
    pending_read_entries() const&& = delete;
    [[nodiscard]] std::span<const WavefrontPathSlot> write_entries() const& noexcept;
    [[nodiscard]] std::span<const WavefrontPathSlot> write_entries() && = delete;
    [[nodiscard]] std::span<const WavefrontPathSlot> write_entries() const&& = delete;

    [[nodiscard]] WavefrontQueuePushStatus push_write(WavefrontPathSlot slot) noexcept;
    [[nodiscard]] WavefrontQueuePushStatus
    push_write_batch(std::span<const WavefrontPathSlot> slots) noexcept;

    // Publishing never clears or replaces an unreleased read generation. The producer may keep
    // filling the separate write buffer after publication, but cannot publish it until release.
    [[nodiscard]] WavefrontQueuePublishStatus publish_write_buffer() noexcept;
    [[nodiscard]] WavefrontQueueReleaseStatus release_read_buffer() noexcept;

  private:
    DoubleBufferedWavefrontQueueT(Queue&& first, Queue&& second) noexcept;

    [[nodiscard]] std::size_t write_index() const noexcept;
    [[nodiscard]] Queue& read_queue() noexcept;
    [[nodiscard]] const Queue& read_queue() const noexcept;
    [[nodiscard]] Queue& write_queue() noexcept;
    [[nodiscard]] const Queue& write_queue() const noexcept;

    std::array<Queue, 2U> queues_;
    std::size_t read_index_{};
    bool read_buffer_pending_{};
};

extern template class BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
extern template class BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::camera>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::ray>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::hit>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::miss>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shade>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shadow>;
extern template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::continuation>;

} // namespace wavefront_queue_detail

using CameraQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
using RayQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
using HitQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
using MissQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
using ShadeQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
using ShadowQueue = wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
using ContinuationQueue =
    wavefront_queue_detail::BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;

using CameraQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::camera>;
using RayQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::ray>;
using HitQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::hit>;
using MissQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::miss>;
using ShadeQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shade>;
using ShadowQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shadow>;
using ContinuationQueueDoubleBuffer =
    wavefront_queue_detail::DoubleBufferedWavefrontQueueT<WavefrontQueueKind::continuation>;

static_assert(std::is_standard_layout_v<WavefrontPathSlot>);
static_assert(std::is_trivially_copyable_v<WavefrontPathSlot>);
static_assert(sizeof(WavefrontPathSlot) == sizeof(std::uint32_t));
static_assert(sizeof(WavefrontQueueKind) == sizeof(std::uint8_t));
static_assert(sizeof(WavefrontQueuePushStatus) == sizeof(std::uint8_t));
static_assert(sizeof(WavefrontQueuePublishStatus) == sizeof(std::uint8_t));
static_assert(sizeof(WavefrontQueueReleaseStatus) == sizeof(std::uint8_t));
static_assert(!std::is_default_constructible_v<CameraQueue>);
static_assert(!std::is_copy_constructible_v<CameraQueue>);
static_assert(std::is_nothrow_move_constructible_v<CameraQueue>);
static_assert(!std::is_default_constructible_v<CameraQueueDoubleBuffer>);
static_assert(!std::is_copy_constructible_v<CameraQueueDoubleBuffer>);
static_assert(std::is_nothrow_move_constructible_v<CameraQueueDoubleBuffer>);

} // namespace blackframe::renderer
