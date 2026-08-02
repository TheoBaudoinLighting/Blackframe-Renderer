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

template class BoundedWavefrontQueueT<WavefrontQueueKind::camera>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::ray>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::hit>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::miss>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::shade>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::shadow>;
template class BoundedWavefrontQueueT<WavefrontQueueKind::continuation>;

} // namespace wavefront_queue_detail
} // namespace blackframe::renderer
