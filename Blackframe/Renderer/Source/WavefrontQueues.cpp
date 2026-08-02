#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace blackframe::renderer {
namespace wavefront_queue_detail {
namespace {

[[nodiscard]] core::Error queue_storage_error(const WavefrontQueueKind kind,
                                              const char* const reason) {
    return core::Error{
        .code = core::StatusCode::resource_exhausted,
        .message = "The " + std::string{wavefront_queue_kind_name(kind)} + " queue " + reason,
    };
}

} // namespace

template <WavefrontQueueKind Kind>
BoundedWavefrontQueueT<Kind>::BoundedWavefrontQueueT(BoundedWavefrontQueueT&& other) noexcept {
    swap(other);
}

template <WavefrontQueueKind Kind>
void BoundedWavefrontQueueT<Kind>::swap(BoundedWavefrontQueueT& other) noexcept {
    using std::swap;
    swap(storage_, other.storage_);
    swap(active_size_, other.active_size_);
}

template <WavefrontQueueKind Kind>
core::Result<BoundedWavefrontQueueT<Kind>>
BoundedWavefrontQueueT<Kind>::create(const std::size_t capacity) {
    if (capacity > std::vector<WavefrontPathSlot>{}.max_size()) {
        return std::unexpected(
            queue_storage_error(Kind, "capacity exceeds the host container limit."));
    }

    auto result = BoundedWavefrontQueueT{};
    try {
        result.storage_.resize(capacity);
    } catch (const std::bad_alloc&) {
        return std::unexpected(queue_storage_error(Kind, "allocation exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            queue_storage_error(Kind, "capacity exceeds the host container limit."));
    }
    return result;
}

template <WavefrontQueueKind Kind>
std::size_t BoundedWavefrontQueueT<Kind>::capacity() const noexcept {
    return storage_.size();
}

template <WavefrontQueueKind Kind> std::size_t BoundedWavefrontQueueT<Kind>::size() const noexcept {
    return active_size_;
}

template <WavefrontQueueKind Kind>
std::size_t BoundedWavefrontQueueT<Kind>::remaining_capacity() const noexcept {
    return capacity() - active_size_;
}

template <WavefrontQueueKind Kind> bool BoundedWavefrontQueueT<Kind>::empty() const noexcept {
    return active_size_ == 0U;
}

template <WavefrontQueueKind Kind> bool BoundedWavefrontQueueT<Kind>::full() const noexcept {
    return active_size_ == capacity();
}

template <WavefrontQueueKind Kind>
std::span<const WavefrontPathSlot> BoundedWavefrontQueueT<Kind>::entries() const& noexcept {
    return {storage_.data(), active_size_};
}

template <WavefrontQueueKind Kind>
WavefrontQueuePushStatus BoundedWavefrontQueueT<Kind>::push(const WavefrontPathSlot slot) noexcept {
    if (full()) {
        return WavefrontQueuePushStatus::capacity_exhausted;
    }
    storage_[active_size_] = slot;
    ++active_size_;
    return WavefrontQueuePushStatus::pushed;
}

template <WavefrontQueueKind Kind>
WavefrontQueuePushStatus
BoundedWavefrontQueueT<Kind>::push_batch(const std::span<const WavefrontPathSlot> slots) noexcept {
    if (slots.size() > remaining_capacity()) {
        return WavefrontQueuePushStatus::capacity_exhausted;
    }
    for (auto index = std::size_t{}; index < slots.size(); ++index) {
        storage_[active_size_ + index] = slots[index];
    }
    active_size_ += slots.size();
    return WavefrontQueuePushStatus::pushed;
}

template <WavefrontQueueKind Kind> void BoundedWavefrontQueueT<Kind>::clear() noexcept {
    active_size_ = 0U;
}

template <WavefrontQueueKind Kind>
DoubleBufferedWavefrontQueueT<Kind>::DoubleBufferedWavefrontQueueT(Queue&& first,
                                                                   Queue&& second) noexcept
    : queues_{std::move(first), std::move(second)} {}

template <WavefrontQueueKind Kind>
DoubleBufferedWavefrontQueueT<Kind>::DoubleBufferedWavefrontQueueT(
    DoubleBufferedWavefrontQueueT&& other) noexcept
    : queues_{std::move(other.queues_[0]), std::move(other.queues_[1])},
      read_index_{std::exchange(other.read_index_, 0U)},
      read_buffer_pending_{std::exchange(other.read_buffer_pending_, false)} {}

template <WavefrontQueueKind Kind>
core::Result<DoubleBufferedWavefrontQueueT<Kind>>
DoubleBufferedWavefrontQueueT<Kind>::create(const std::size_t capacity) {
    auto first = Queue::create(capacity);
    if (!first.has_value()) {
        return std::unexpected(std::move(first.error()));
    }

    auto second = Queue::create(capacity);
    if (!second.has_value()) {
        return std::unexpected(std::move(second.error()));
    }

    return DoubleBufferedWavefrontQueueT{std::move(*first), std::move(*second)};
}

template <WavefrontQueueKind Kind>
std::size_t DoubleBufferedWavefrontQueueT<Kind>::capacity() const noexcept {
    return queues_[0].capacity();
}

template <WavefrontQueueKind Kind>
std::size_t DoubleBufferedWavefrontQueueT<Kind>::write_size() const noexcept {
    return write_queue().size();
}

template <WavefrontQueueKind Kind>
std::size_t DoubleBufferedWavefrontQueueT<Kind>::write_remaining_capacity() const noexcept {
    return write_queue().remaining_capacity();
}

template <WavefrontQueueKind Kind>
bool DoubleBufferedWavefrontQueueT<Kind>::has_pending_read() const noexcept {
    return read_buffer_pending_;
}

template <WavefrontQueueKind Kind>
bool DoubleBufferedWavefrontQueueT<Kind>::write_empty() const noexcept {
    return write_queue().empty();
}

template <WavefrontQueueKind Kind>
bool DoubleBufferedWavefrontQueueT<Kind>::write_full() const noexcept {
    return write_queue().full();
}

template <WavefrontQueueKind Kind>
std::optional<std::span<const WavefrontPathSlot>>
DoubleBufferedWavefrontQueueT<Kind>::pending_read_entries() const& noexcept {
    if (!read_buffer_pending_) {
        return std::nullopt;
    }
    return read_queue().entries();
}

template <WavefrontQueueKind Kind>
std::span<const WavefrontPathSlot>
DoubleBufferedWavefrontQueueT<Kind>::write_entries() const& noexcept {
    return write_queue().entries();
}

template <WavefrontQueueKind Kind>
WavefrontQueuePushStatus
DoubleBufferedWavefrontQueueT<Kind>::push_write(const WavefrontPathSlot slot) noexcept {
    return write_queue().push(slot);
}

template <WavefrontQueueKind Kind>
WavefrontQueuePushStatus DoubleBufferedWavefrontQueueT<Kind>::push_write_batch(
    const std::span<const WavefrontPathSlot> slots) noexcept {
    return write_queue().push_batch(slots);
}

template <WavefrontQueueKind Kind>
WavefrontQueuePublishStatus DoubleBufferedWavefrontQueueT<Kind>::publish_write_buffer() noexcept {
    if (read_buffer_pending_) {
        return WavefrontQueuePublishStatus::read_buffer_pending;
    }

    read_index_ = write_index();
    read_buffer_pending_ = true;
    return WavefrontQueuePublishStatus::published;
}

template <WavefrontQueueKind Kind>
WavefrontQueueReleaseStatus DoubleBufferedWavefrontQueueT<Kind>::release_read_buffer() noexcept {
    if (!read_buffer_pending_) {
        return WavefrontQueueReleaseStatus::no_read_buffer_pending;
    }

    read_queue().clear();
    read_buffer_pending_ = false;
    return WavefrontQueueReleaseStatus::released;
}

template <WavefrontQueueKind Kind>
std::size_t DoubleBufferedWavefrontQueueT<Kind>::write_index() const noexcept {
    return 1U - read_index_;
}

template <WavefrontQueueKind Kind>
typename DoubleBufferedWavefrontQueueT<Kind>::Queue&
DoubleBufferedWavefrontQueueT<Kind>::read_queue() noexcept {
    return queues_[read_index_];
}

template <WavefrontQueueKind Kind>
const typename DoubleBufferedWavefrontQueueT<Kind>::Queue&
DoubleBufferedWavefrontQueueT<Kind>::read_queue() const noexcept {
    return queues_[read_index_];
}

template <WavefrontQueueKind Kind>
typename DoubleBufferedWavefrontQueueT<Kind>::Queue&
DoubleBufferedWavefrontQueueT<Kind>::write_queue() noexcept {
    return queues_[write_index()];
}

template <WavefrontQueueKind Kind>
const typename DoubleBufferedWavefrontQueueT<Kind>::Queue&
DoubleBufferedWavefrontQueueT<Kind>::write_queue() const noexcept {
    return queues_[write_index()];
}

template class BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::camera>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::ray>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::hit>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::miss>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shade>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::shadow>;
template class DoubleBufferedWavefrontQueueT<WavefrontQueueKind::continuation>;

} // namespace wavefront_queue_detail
} // namespace blackframe::renderer
