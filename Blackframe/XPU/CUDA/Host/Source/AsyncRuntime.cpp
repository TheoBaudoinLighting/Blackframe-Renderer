#include <Blackframe/XPU/CUDA/AsyncRuntime.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace blackframe::xpu::cuda {
namespace {

[[nodiscard]] core::StatusCode runtime_status_code(const cudaError_t status) noexcept {
    if (status == cudaSuccess) {
        return core::StatusCode::success;
    }
    if (status == cudaErrorMemoryAllocation) {
        return core::StatusCode::resource_exhausted;
    }
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        status == cudaErrorInitializationError) {
        return core::StatusCode::unavailable;
    }
    return core::StatusCode::platform_error;
}

[[nodiscard]] core::Error runtime_error(const cudaError_t status,
                                        const std::string_view operation) {
    return core::Error{
        .code = runtime_status_code(status),
        .message = "CUDA " + std::string{operation} + " failed: " + cudaGetErrorName(status) +
                   " (" + cudaGetErrorString(status) + ").",
    };
}

[[nodiscard]] core::Error ownership_error(const std::string_view resource) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message =
            "CUDA " + std::string{resource} + " operation requires its owning device to be active.",
    };
}

[[nodiscard]] core::Error timing_not_ready_error(const std::string_view event_name) {
    return core::Error{
        .code = core::StatusCode::unavailable,
        .message =
            "CUDA timing " + std::string{event_name} + " event is not ready: cudaErrorNotReady.",
    };
}

[[nodiscard]] core::Error timing_call_order_error(const std::string_view operation,
                                                  const std::string_view requirement) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "CUDA timing event pair " + std::string{operation} + " requires " +
                   std::string{requirement} + ".",
    };
}

[[nodiscard]] core::Status require_active_device(const std::int32_t device_ordinal,
                                                 const std::string_view resource) {
    auto active_device = int{-1};
    const auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, std::string{resource} + " device query"));
    }
    if (device_ordinal < 0 || active_device != device_ordinal) {
        return std::unexpected(ownership_error(resource));
    }
    return {};
}

struct DeviceActivation final {
    std::int32_t previous_device{-1};
    bool changed{};
};

[[nodiscard]] core::Result<DeviceActivation>
activate_owning_device(const std::int32_t device_ordinal, const std::string_view resource) {
    auto active_device = int{-1};
    auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, std::string{resource} + " device query"));
    }
    if (device_ordinal < 0) {
        return std::unexpected(ownership_error(resource));
    }
    const auto changed = active_device != device_ordinal;
    if (changed) {
        status = cudaSetDevice(device_ordinal);
        if (status != cudaSuccess) {
            return std::unexpected(
                runtime_error(status, std::string{resource} + " device activation"));
        }
    }
    return DeviceActivation{.previous_device = active_device, .changed = changed};
}

[[nodiscard]] core::Status restore_device(const DeviceActivation activation,
                                          const std::string_view resource) {
    if (!activation.changed) {
        return {};
    }
    const auto status = cudaSetDevice(activation.previous_device);
    if (status != cudaSuccess) {
        return std::unexpected(
            runtime_error(status, std::string{resource} + " device restoration"));
    }
    return {};
}

[[nodiscard]] cudaStream_t native_stream(const StreamHandle handle) noexcept {
    return static_cast<cudaStream_t>(handle);
}

[[nodiscard]] cudaEvent_t native_event(const EventHandle handle) noexcept {
    return static_cast<cudaEvent_t>(handle);
}

[[nodiscard]] core::Result<cudaStream_t> timing_stream(const Stream* const stream,
                                                       const std::int32_t device_ordinal,
                                                       const std::string_view operation) {
    if (stream == nullptr) {
        return cudaStream_t{};
    }
    if (!static_cast<bool>(*stream)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "CUDA timing event pair " + std::string{operation} +
                       " requires an open stream or an explicit nullptr default stream.",
        });
    }
    if (stream->device_ordinal() != device_ordinal) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "CUDA timing event pair cannot " + std::string{operation} +
                       " on a stream owned by another device.",
        });
    }
    return native_stream(stream->native_handle());
}

[[nodiscard]] core::Status require_timing_event_complete(const cudaEvent_t event,
                                                         const std::string_view event_name) {
    const auto status = cudaEventQuery(event);
    if (status == cudaSuccess) {
        return {};
    }
    if (status == cudaErrorNotReady) {
        return std::unexpected(timing_not_ready_error(event_name));
    }
    return std::unexpected(
        runtime_error(status, "timing " + std::string{event_name} + " event query"));
}

[[nodiscard]] core::Result<std::uint64_t> nanoseconds_from_milliseconds(const float milliseconds) {
    constexpr auto nanoseconds_per_millisecond = 1'000'000.0;
    constexpr auto uint64_limit_exclusive = 18'446'744'073'709'551'616.0;

    if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::platform_error,
            .message = "CUDA timing elapsed milliseconds are not finite and non-negative.",
        });
    }
    const auto nanoseconds = static_cast<double>(milliseconds) * nanoseconds_per_millisecond;
    const auto rounded_nanoseconds = std::round(nanoseconds);
    if (!std::isfinite(rounded_nanoseconds) || rounded_nanoseconds < 0.0 ||
        rounded_nanoseconds >= uint64_limit_exclusive) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "CUDA timing elapsed nanoseconds exceed the uint64 range.",
        });
    }
    return static_cast<std::uint64_t>(rounded_nanoseconds);
}

} // namespace

Stream::Stream(const StreamHandle handle, const std::int32_t device_ordinal) noexcept
    : handle_(handle), device_ordinal_(device_ordinal) {}

Stream::~Stream() noexcept {
    close_without_diagnostic();
}

Stream::Stream(Stream&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)) {}

core::Result<Stream> Stream::create() {
    auto device_ordinal = int{-1};
    auto status = cudaGetDevice(&device_ordinal);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "nonblocking stream device query"));
    }

    auto stream = cudaStream_t{};
    status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "nonblocking stream creation"));
    }
    if (stream == nullptr) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message = "CUDA stream creation succeeded without returning a native handle.",
        });
    }
    return Stream{static_cast<void*>(stream), device_ordinal};
}

core::Status Stream::wait(const Event& event) const {
    if (empty() || event.empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "CUDA stream wait requires open stream and event handles.",
        });
    }
    if (device_ordinal_ != event.device_ordinal_) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "CUDA stream cannot wait for an event owned by another device.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "stream wait"); !active) {
        return active;
    }
    const auto status =
        cudaStreamWaitEvent(native_stream(handle_), native_event(event.handle_), 0U);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "stream event wait"));
    }
    return {};
}

core::Status Stream::synchronize() const {
    if (empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot synchronize a closed CUDA stream.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "stream synchronization"); !active) {
        return active;
    }
    const auto status = cudaStreamSynchronize(native_stream(handle_));
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "stream synchronization"));
    }
    return {};
}

core::Status Stream::close() {
    if (empty()) {
        return {};
    }
    auto activation = activate_owning_device(device_ordinal_, "stream close");
    if (!activation) {
        return std::unexpected(std::move(activation.error()));
    }

    const auto synchronization_status = cudaStreamSynchronize(native_stream(handle_));
    const auto destroy_status = cudaStreamDestroy(native_stream(handle_));
    if (destroy_status == cudaSuccess) {
        handle_ = nullptr;
        device_ordinal_ = -1;
    }
    auto restoration_status = restore_device(*activation, "stream close");

    if (synchronization_status != cudaSuccess) {
        return std::unexpected(
            runtime_error(synchronization_status, "stream close synchronization"));
    }
    if (destroy_status != cudaSuccess) {
        return std::unexpected(runtime_error(destroy_status, "stream destruction"));
    }
    if (!restoration_status) {
        return restoration_status;
    }
    return {};
}

void Stream::close_without_diagnostic() noexcept {
    if (empty()) {
        return;
    }
    auto previous_device = int{-1};
    const auto get_status = cudaGetDevice(&previous_device);
    const auto changed = get_status == cudaSuccess && previous_device != device_ordinal_;
    const auto set_status = changed ? cudaSetDevice(device_ordinal_) : cudaSuccess;
    if (get_status == cudaSuccess && set_status == cudaSuccess) {
        static_cast<void>(cudaStreamSynchronize(native_stream(handle_)));
        static_cast<void>(cudaStreamDestroy(native_stream(handle_)));
    }
    if (changed && set_status == cudaSuccess) {
        static_cast<void>(cudaSetDevice(previous_device));
    }
    handle_ = nullptr;
    device_ordinal_ = -1;
}

Event::Event(const EventHandle handle, const std::int32_t device_ordinal) noexcept
    : handle_(handle), device_ordinal_(device_ordinal) {}

Event::~Event() noexcept {
    close_without_diagnostic();
}

Event::Event(Event&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)) {}

core::Result<Event> Event::create() {
    auto device_ordinal = int{-1};
    auto status = cudaGetDevice(&device_ordinal);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "dependency event device query"));
    }

    auto event = cudaEvent_t{};
    status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "dependency event creation"));
    }
    if (event == nullptr) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message = "CUDA event creation succeeded without returning a native handle.",
        });
    }
    return Event{static_cast<void*>(event), device_ordinal};
}

core::Status Event::record(const Stream& stream) const {
    if (empty() || stream.empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "CUDA event recording requires open event and stream handles.",
        });
    }
    if (device_ordinal_ != stream.device_ordinal()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "CUDA event cannot be recorded on a stream owned by another device.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "event recording"); !active) {
        return active;
    }
    const auto status =
        cudaEventRecord(native_event(handle_), native_stream(stream.native_handle()));
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "event recording"));
    }
    return {};
}

core::Status Event::synchronize() const {
    if (empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot synchronize a closed CUDA event.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "event synchronization"); !active) {
        return active;
    }
    const auto status = cudaEventSynchronize(native_event(handle_));
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "event synchronization"));
    }
    return {};
}

core::Result<bool> Event::is_complete() const {
    if (empty()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot query a closed CUDA event.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "event query"); !active) {
        return std::unexpected(std::move(active.error()));
    }
    const auto status = cudaEventQuery(native_event(handle_));
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        return false;
    }
    return std::unexpected(runtime_error(status, "event query"));
}

core::Status Event::close() {
    if (empty()) {
        return {};
    }
    auto activation = activate_owning_device(device_ordinal_, "event close");
    if (!activation) {
        return std::unexpected(std::move(activation.error()));
    }

    const auto synchronization_status = cudaEventSynchronize(native_event(handle_));
    const auto destroy_status = cudaEventDestroy(native_event(handle_));
    if (destroy_status == cudaSuccess) {
        handle_ = nullptr;
        device_ordinal_ = -1;
    }
    auto restoration_status = restore_device(*activation, "event close");

    if (synchronization_status != cudaSuccess) {
        return std::unexpected(
            runtime_error(synchronization_status, "event close synchronization"));
    }
    if (destroy_status != cudaSuccess) {
        return std::unexpected(runtime_error(destroy_status, "event destruction"));
    }
    if (!restoration_status) {
        return restoration_status;
    }
    return {};
}

void Event::close_without_diagnostic() noexcept {
    if (empty()) {
        return;
    }
    auto previous_device = int{-1};
    const auto get_status = cudaGetDevice(&previous_device);
    const auto changed = get_status == cudaSuccess && previous_device != device_ordinal_;
    const auto set_status = changed ? cudaSetDevice(device_ordinal_) : cudaSuccess;
    if (get_status == cudaSuccess && set_status == cudaSuccess) {
        static_cast<void>(cudaEventSynchronize(native_event(handle_)));
        static_cast<void>(cudaEventDestroy(native_event(handle_)));
    }
    if (changed && set_status == cudaSuccess) {
        static_cast<void>(cudaSetDevice(previous_device));
    }
    handle_ = nullptr;
    device_ordinal_ = -1;
}

TimingEventPair::TimingEventPair(const EventHandle begin_handle, const EventHandle end_handle,
                                 const std::int32_t device_ordinal) noexcept
    : begin_handle_(begin_handle), end_handle_(end_handle), device_ordinal_(device_ordinal) {}

TimingEventPair::~TimingEventPair() noexcept {
    close_without_diagnostic();
}

TimingEventPair::TimingEventPair(TimingEventPair&& other) noexcept
    : begin_handle_(std::exchange(other.begin_handle_, nullptr)),
      end_handle_(std::exchange(other.end_handle_, nullptr)),
      recorded_stream_(std::exchange(other.recorded_stream_, nullptr)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)),
      state_(std::exchange(other.state_, State::ready)) {}

core::Result<TimingEventPair> TimingEventPair::create() {
    auto device_ordinal = int{-1};
    auto status = cudaGetDevice(&device_ordinal);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "timing event pair device query"));
    }

    auto begin_event = cudaEvent_t{};
    status = cudaEventCreateWithFlags(&begin_event, cudaEventDefault);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "timing begin event creation"));
    }
    if (begin_event == nullptr) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message =
                "CUDA timing begin event creation succeeded without returning a native handle.",
        });
    }

    auto end_event = cudaEvent_t{};
    status = cudaEventCreateWithFlags(&end_event, cudaEventDefault);
    if (status != cudaSuccess || end_event == nullptr) {
        auto error = status != cudaSuccess
                         ? runtime_error(status, "timing end event creation")
                         : core::Error{
                               .code = core::StatusCode::internal_error,
                               .message = "CUDA timing end event creation succeeded without "
                                          "returning a native handle.",
                           };
        const auto end_cleanup_status =
            end_event != nullptr ? cudaEventDestroy(end_event) : cudaSuccess;
        const auto begin_cleanup_status = cudaEventDestroy(begin_event);
        if (end_cleanup_status != cudaSuccess) {
            error.message +=
                " Cleanup also failed: " +
                runtime_error(end_cleanup_status, "timing end event destruction").message;
        }
        if (begin_cleanup_status != cudaSuccess) {
            error.message +=
                " Cleanup also failed: " +
                runtime_error(begin_cleanup_status, "timing begin event destruction").message;
        }
        return std::unexpected(std::move(error));
    }

    return TimingEventPair{static_cast<void*>(begin_event), static_cast<void*>(end_event),
                           device_ordinal};
}

core::Status TimingEventPair::begin(const Stream* const stream) {
    if (!static_cast<bool>(*this)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot begin a closed or partially closed CUDA timing event pair.",
        });
    }
    if (state_ != State::ready) {
        return std::unexpected(timing_call_order_error(
            "begin", "a fresh pair or a successfully consumed elapsed interval"));
    }
    auto selected_stream = timing_stream(stream, device_ordinal_, "begin");
    if (!selected_stream) {
        return std::unexpected(std::move(selected_stream.error()));
    }
    if (auto active = require_active_device(device_ordinal_, "timing begin recording"); !active) {
        return active;
    }
    const auto status = cudaEventRecord(native_event(begin_handle_), *selected_stream);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "timing begin event recording"));
    }
    recorded_stream_ = static_cast<void*>(*selected_stream);
    state_ = State::begun;
    return {};
}

core::Status TimingEventPair::end(const Stream* const stream) {
    if (!static_cast<bool>(*this)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot end a closed or partially closed CUDA timing event pair.",
        });
    }
    if (state_ != State::begun) {
        return std::unexpected(timing_call_order_error(
            "end", "exactly one successful begin for the current interval"));
    }
    auto selected_stream = timing_stream(stream, device_ordinal_, "end");
    if (!selected_stream) {
        return std::unexpected(std::move(selected_stream.error()));
    }
    if (static_cast<void*>(*selected_stream) != recorded_stream_) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::incompatible,
            .message = "CUDA timing event pair begin and end must use the same stream.",
        });
    }
    if (auto active = require_active_device(device_ordinal_, "timing end recording"); !active) {
        return active;
    }
    const auto status = cudaEventRecord(native_event(end_handle_), *selected_stream);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "timing end event recording"));
    }
    state_ = State::ended;
    return {};
}

core::Result<std::uint64_t> TimingEventPair::elapsed_nanoseconds() {
    if (!static_cast<bool>(*this)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Cannot query a closed or partially closed CUDA timing event pair.",
        });
    }
    if (state_ != State::ended) {
        return std::unexpected(timing_call_order_error(
            "elapsed-time query", "a successful begin and end for the current interval"));
    }
    if (auto active = require_active_device(device_ordinal_, "timing elapsed query"); !active) {
        return std::unexpected(std::move(active.error()));
    }
    if (auto complete = require_timing_event_complete(native_event(begin_handle_), "begin");
        !complete) {
        return std::unexpected(std::move(complete.error()));
    }
    if (auto complete = require_timing_event_complete(native_event(end_handle_), "end");
        !complete) {
        return std::unexpected(std::move(complete.error()));
    }

    auto elapsed_milliseconds = float{};
    const auto status = cudaEventElapsedTime(&elapsed_milliseconds, native_event(begin_handle_),
                                             native_event(end_handle_));
    if (status == cudaErrorNotReady) {
        return std::unexpected(timing_not_ready_error("elapsed"));
    }
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "timing elapsed-time query"));
    }
    auto elapsed_nanoseconds = nanoseconds_from_milliseconds(elapsed_milliseconds);
    if (!elapsed_nanoseconds) {
        return std::unexpected(std::move(elapsed_nanoseconds.error()));
    }
    recorded_stream_ = nullptr;
    state_ = State::ready;
    return *elapsed_nanoseconds;
}

core::Status TimingEventPair::close() {
    if (empty()) {
        return {};
    }
    auto activation = activate_owning_device(device_ordinal_, "timing event pair close");
    if (!activation) {
        return std::unexpected(std::move(activation.error()));
    }

    const auto begin_destroy_status =
        begin_handle_ != nullptr ? cudaEventDestroy(native_event(begin_handle_)) : cudaSuccess;
    if (begin_destroy_status == cudaSuccess) {
        begin_handle_ = nullptr;
    }
    const auto end_destroy_status =
        end_handle_ != nullptr ? cudaEventDestroy(native_event(end_handle_)) : cudaSuccess;
    if (end_destroy_status == cudaSuccess) {
        end_handle_ = nullptr;
    }
    if (empty()) {
        recorded_stream_ = nullptr;
        device_ordinal_ = -1;
        state_ = State::ready;
    }
    auto restoration_status = restore_device(*activation, "timing event pair close");

    if (begin_destroy_status != cudaSuccess) {
        auto error = runtime_error(begin_destroy_status, "timing begin event destruction");
        if (end_destroy_status != cudaSuccess) {
            error.message +=
                " Additional failure: " +
                runtime_error(end_destroy_status, "timing end event destruction").message;
        }
        if (!restoration_status) {
            error.message += " Additional failure: " + restoration_status.error().message;
        }
        return std::unexpected(std::move(error));
    }
    if (end_destroy_status != cudaSuccess) {
        auto error = runtime_error(end_destroy_status, "timing end event destruction");
        if (!restoration_status) {
            error.message += " Additional failure: " + restoration_status.error().message;
        }
        return std::unexpected(std::move(error));
    }
    if (!restoration_status) {
        return restoration_status;
    }
    return {};
}

void TimingEventPair::close_without_diagnostic() noexcept {
    if (empty()) {
        return;
    }
    auto previous_device = int{-1};
    const auto get_status = cudaGetDevice(&previous_device);
    const auto changed = get_status == cudaSuccess && previous_device != device_ordinal_;
    const auto set_status = changed ? cudaSetDevice(device_ordinal_) : cudaSuccess;
    if (get_status == cudaSuccess && set_status == cudaSuccess) {
        if (begin_handle_ != nullptr) {
            static_cast<void>(cudaEventDestroy(native_event(begin_handle_)));
        }
        if (end_handle_ != nullptr) {
            static_cast<void>(cudaEventDestroy(native_event(end_handle_)));
        }
    }
    if (changed && set_status == cudaSuccess) {
        static_cast<void>(cudaSetDevice(previous_device));
    }
    begin_handle_ = nullptr;
    end_handle_ = nullptr;
    recorded_stream_ = nullptr;
    device_ordinal_ = -1;
    state_ = State::ready;
}

PinnedHostAllocation::PinnedHostAllocation(void* const data, const std::size_t size_bytes,
                                           const std::int32_t device_ordinal) noexcept
    : data_(data), size_bytes_(size_bytes), device_ordinal_(device_ordinal) {}

PinnedHostAllocation::~PinnedHostAllocation() noexcept {
    close_without_diagnostic();
}

PinnedHostAllocation::PinnedHostAllocation(PinnedHostAllocation&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_bytes_(std::exchange(other.size_bytes_, 0U)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)) {}

core::Result<PinnedHostAllocation>
PinnedHostAllocation::allocate_bytes(const std::size_t byte_count,
                                     const PinnedHostMemoryBudget budget) {
    if (byte_count > budget.maximum_bytes) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "CUDA pinned-host allocation exceeds its explicit memory budget.",
        });
    }
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::resource_exhausted,
            .message = "CUDA pinned-host allocation exceeds the addressable range.",
        });
    }
    if (byte_count == 0U) {
        return PinnedHostAllocation{};
    }

    auto device_ordinal = int{-1};
    auto status = cudaGetDevice(&device_ordinal);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "pinned-host allocation device query"));
    }
    void* data = nullptr;
    status = cudaHostAlloc(&data, byte_count, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, "pinned-host allocation"));
    }
    if (data == nullptr) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message = "CUDA pinned-host allocation succeeded without returning storage.",
        });
    }
    return PinnedHostAllocation{data, byte_count, device_ordinal};
}

core::Status PinnedHostAllocation::close() {
    if (empty()) {
        return {};
    }
    auto activation = activate_owning_device(device_ordinal_, "pinned-host allocation close");
    if (!activation) {
        return std::unexpected(std::move(activation.error()));
    }

    const auto free_status = cudaFreeHost(data_);
    if (free_status == cudaSuccess) {
        data_ = nullptr;
        size_bytes_ = 0U;
        device_ordinal_ = -1;
    }
    auto restoration_status = restore_device(*activation, "pinned-host allocation close");

    if (free_status != cudaSuccess) {
        return std::unexpected(runtime_error(free_status, "pinned-host storage release"));
    }
    if (!restoration_status) {
        return restoration_status;
    }
    return {};
}

void PinnedHostAllocation::close_without_diagnostic() noexcept {
    if (empty()) {
        return;
    }
    auto previous_device = int{-1};
    const auto get_status = cudaGetDevice(&previous_device);
    const auto changed = get_status == cudaSuccess && previous_device != device_ordinal_;
    const auto set_status = changed ? cudaSetDevice(device_ordinal_) : cudaSuccess;
    if (get_status == cudaSuccess && set_status == cudaSuccess) {
        static_cast<void>(cudaFreeHost(data_));
    }
    if (changed && set_status == cudaSuccess) {
        static_cast<void>(cudaSetDevice(previous_device));
    }
    data_ = nullptr;
    size_bytes_ = 0U;
    device_ordinal_ = -1;
}

} // namespace blackframe::xpu::cuda
