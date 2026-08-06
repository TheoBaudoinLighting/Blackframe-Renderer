#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace blackframe::xpu::cuda {

// CUDA runtime types intentionally stay out of this public host header. The
// native handles are opaque and are only interpreted by CUDA-owning source
// files that include cuda_runtime_api.h.
using StreamHandle = void*;
using EventHandle = void*;

class Event;

class Stream final {
  public:
    Stream() noexcept = default;
    ~Stream() noexcept;

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) = delete;

    // Streams created by this API always use cudaStreamNonBlocking. No default-
    // stream substitution is performed when creation fails.
    [[nodiscard]] static core::Result<Stream> create();

    [[nodiscard]] core::Status wait(const Event& event) const;
    [[nodiscard]] core::Status synchronize() const;

    // Explicit close waits for queued work before destroying the stream and
    // reports both execution and teardown failures. The destructor is only the
    // no-throw ownership backstop.
    [[nodiscard]] core::Status close();

    [[nodiscard]] StreamHandle native_handle() const noexcept {
        return handle_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_ordinal_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return handle_ == nullptr;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !empty();
    }

  private:
    Stream(StreamHandle handle, std::int32_t device_ordinal) noexcept;
    void close_without_diagnostic() noexcept;

    StreamHandle handle_{nullptr};
    std::int32_t device_ordinal_{-1};
};

class Event final {
  public:
    Event() noexcept = default;
    ~Event() noexcept;

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) = delete;

    // Dependency events disable timing so their contract is ordering only.
    [[nodiscard]] static core::Result<Event> create();

    [[nodiscard]] core::Status record(const Stream& stream) const;
    [[nodiscard]] core::Status synchronize() const;
    [[nodiscard]] core::Result<bool> is_complete() const;
    [[nodiscard]] core::Status close();

    [[nodiscard]] EventHandle native_handle() const noexcept {
        return handle_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_ordinal_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return handle_ == nullptr;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !empty();
    }

  private:
    friend class Stream;

    Event(EventHandle handle, std::int32_t device_ordinal) noexcept;
    void close_without_diagnostic() noexcept;

    EventHandle handle_{nullptr};
    std::int32_t device_ordinal_{-1};
};

class TimingEventPair final {
  public:
    TimingEventPair() noexcept = default;
    ~TimingEventPair() noexcept;

    TimingEventPair(const TimingEventPair&) = delete;
    TimingEventPair& operator=(const TimingEventPair&) = delete;

    TimingEventPair(TimingEventPair&& other) noexcept;
    TimingEventPair& operator=(TimingEventPair&& other) = delete;

    // Timing events deliberately remain separate from dependency Event. Passing
    // nullptr explicitly selects CUDA's default stream; a non-null pointer must
    // refer to an open stream owned by the same device. Both records for one
    // interval must select the same stream.
    [[nodiscard]] static core::Result<TimingEventPair> create();
    [[nodiscard]] core::Status begin(const Stream* stream);
    [[nodiscard]] core::Status end(const Stream* stream);

    // This is a non-blocking query. A pending event is reported as an explicit
    // cudaErrorNotReady-derived error and never synchronized here. A successful
    // query consumes the completed interval and rearms the pair for begin().
    [[nodiscard]] core::Result<std::uint64_t> elapsed_nanoseconds();

    [[nodiscard]] core::Status close();

    [[nodiscard]] EventHandle begin_native_handle() const noexcept {
        return begin_handle_;
    }
    [[nodiscard]] EventHandle end_native_handle() const noexcept {
        return end_handle_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_ordinal_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return begin_handle_ == nullptr && end_handle_ == nullptr;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return begin_handle_ != nullptr && end_handle_ != nullptr;
    }

  private:
    enum class State : std::uint8_t {
        ready,
        begun,
        ended,
    };

    TimingEventPair(EventHandle begin_handle, EventHandle end_handle,
                    std::int32_t device_ordinal) noexcept;
    void close_without_diagnostic() noexcept;

    EventHandle begin_handle_{nullptr};
    EventHandle end_handle_{nullptr};
    StreamHandle recorded_stream_{nullptr};
    std::int32_t device_ordinal_{-1};
    State state_{State::ready};
};

struct PinnedHostMemoryBudget final {
    std::size_t maximum_bytes{static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())};
};

class PinnedHostAllocation final {
  public:
    PinnedHostAllocation() noexcept = default;
    ~PinnedHostAllocation() noexcept;

    PinnedHostAllocation(const PinnedHostAllocation&) = delete;
    PinnedHostAllocation& operator=(const PinnedHostAllocation&) = delete;

    PinnedHostAllocation(PinnedHostAllocation&& other) noexcept;
    PinnedHostAllocation& operator=(PinnedHostAllocation&& other) = delete;

    [[nodiscard]] static core::Result<PinnedHostAllocation>
    allocate_bytes(std::size_t byte_count, PinnedHostMemoryBudget budget = {});

    // The caller must first complete every asynchronous transfer that refers to
    // this allocation. CUDA cannot infer those stream dependencies from a host
    // pointer, so freeing in-flight storage is deliberately not hidden here.
    [[nodiscard]] core::Status close();

    [[nodiscard]] void* data() noexcept {
        return data_;
    }
    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return size_bytes_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_ordinal_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return data_ == nullptr;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !empty();
    }

  private:
    PinnedHostAllocation(void* data, std::size_t size_bytes, std::int32_t device_ordinal) noexcept;
    void close_without_diagnostic() noexcept;

    void* data_{nullptr};
    std::size_t size_bytes_{0};
    std::int32_t device_ordinal_{-1};
};

template <typename Element>
concept PinnedHostBufferElement =
    std::is_trivially_copyable_v<Element> && std::is_nothrow_default_constructible_v<Element> &&
    !std::is_const_v<Element> && !std::is_volatile_v<Element>;

template <PinnedHostBufferElement Element> class PinnedHostBuffer final {
  public:
    PinnedHostBuffer() noexcept = default;

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : allocation_(std::move(other.allocation_)), data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0U)) {}
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) = delete;

    [[nodiscard]] static core::Result<PinnedHostBuffer>
    allocate(const std::size_t count, const PinnedHostMemoryBudget budget = {}) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::resource_exhausted,
                .message = "CUDA pinned-host buffer size multiplication overflowed.",
            });
        }
        if (count == 0U) {
            return PinnedHostBuffer{};
        }

        const auto payload_bytes = count * sizeof(Element);
        constexpr auto alignment_padding = alignof(Element) - 1U;
        if (payload_bytes > std::numeric_limits<std::size_t>::max() - alignment_padding) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::resource_exhausted,
                .message = "CUDA pinned-host buffer alignment padding overflowed.",
            });
        }

        auto allocation =
            PinnedHostAllocation::allocate_bytes(payload_bytes + alignment_padding, budget);
        if (!allocation) {
            return std::unexpected(std::move(allocation.error()));
        }
        auto* aligned_data = allocation->data();
        auto available_bytes = allocation->size_bytes();
        if (std::align(alignof(Element), payload_bytes, aligned_data, available_bytes) == nullptr) {
            auto close_status = allocation->close();
            auto message = std::string{"CUDA pinned-host allocation could not satisfy the typed "
                                       "buffer alignment contract."};
            if (!close_status) {
                message += " Cleanup also failed: " + close_status.error().message;
            }
            return std::unexpected(core::Error{
                .code = core::StatusCode::internal_error,
                .message = std::move(message),
            });
        }
        static_cast<void>(::new (aligned_data) Element[count]);
        return PinnedHostBuffer{std::move(*allocation), static_cast<Element*>(aligned_data), count};
    }

    [[nodiscard]] core::Status close() {
        auto status = allocation_.close();
        if (allocation_.empty()) {
            data_ = nullptr;
            count_ = 0U;
        }
        return status;
    }

    [[nodiscard]] Element* data() noexcept {
        return data_;
    }
    [[nodiscard]] const Element* data() const noexcept {
        return data_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return count_ * sizeof(Element);
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return allocation_.device_ordinal();
    }
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0U;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !allocation_.empty();
    }

    [[nodiscard]] Element& operator[](const std::size_t index) noexcept {
        return data()[index];
    }
    [[nodiscard]] const Element& operator[](const std::size_t index) const noexcept {
        return data()[index];
    }

  private:
    PinnedHostBuffer(PinnedHostAllocation allocation, Element* const data,
                     const std::size_t count) noexcept
        : allocation_(std::move(allocation)), data_(data), count_(count) {}

    PinnedHostAllocation allocation_{};
    Element* data_{nullptr};
    std::size_t count_{0U};
};

static_assert(std::is_nothrow_move_constructible_v<Stream>);
static_assert(std::is_nothrow_destructible_v<Stream>);
static_assert(std::is_nothrow_move_constructible_v<Event>);
static_assert(std::is_nothrow_destructible_v<Event>);
static_assert(std::is_nothrow_move_constructible_v<TimingEventPair>);
static_assert(std::is_nothrow_destructible_v<TimingEventPair>);
static_assert(std::is_nothrow_move_constructible_v<PinnedHostAllocation>);
static_assert(std::is_nothrow_destructible_v<PinnedHostAllocation>);

} // namespace blackframe::xpu::cuda
