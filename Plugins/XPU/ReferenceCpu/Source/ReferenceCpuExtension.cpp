#include <Blackframe/Extensions/ExtensionAbi.h>
#include <Blackframe/XPU/XpuBackendAbi.h>
#include <stdint.h>

namespace {

constexpr char backend_name[] = "Reference CPU";
constexpr char device_name[] = "Host CPU";
constexpr char vendor_name[] = "Blackframe";

template <uint32_t Size>
[[nodiscard]] constexpr auto make_view(const char (&text)[Size]) noexcept -> blackframe_utf8_view {
    static_assert(Size > 0);
    return blackframe_utf8_view{
        .data = text,
        .size = Size - 1,
        .reserved = 0,
    };
}

[[nodiscard]] constexpr auto is_xpu_interface(const blackframe_interface_id interface_id) noexcept
    -> bool {
    return interface_id.high == BLACKFRAME_XPU_BACKEND_INTERFACE_ID_HIGH &&
           interface_id.low == BLACKFRAME_XPU_BACKEND_INTERFACE_ID_LOW;
}

auto BLACKFRAME_ABI_CALL get_device_count(void*, uint32_t* const out_device_count) noexcept
    -> blackframe_status {
    if (out_device_count == nullptr) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }
    *out_device_count = 1;
    return BLACKFRAME_STATUS_SUCCESS;
}

auto BLACKFRAME_ABI_CALL get_device_descriptor(
    void*, const uint32_t device_index,
    blackframe_xpu_device_descriptor_v1* const out_descriptor) noexcept -> blackframe_status {
    if (out_descriptor == nullptr ||
        out_descriptor->struct_size < sizeof(blackframe_xpu_device_descriptor_v1)) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }
    if (device_index != 0) {
        return BLACKFRAME_STATUS_OUT_OF_RANGE;
    }

    *out_descriptor = blackframe_xpu_device_descriptor_v1{
        .struct_size = sizeof(blackframe_xpu_device_descriptor_v1),
        .kind = BLACKFRAME_XPU_DEVICE_KIND_CPU,
        .identifier_high = UINT64_C(0xd14f39ae428b46d7),
        .identifier_low = UINT64_C(0xa165a0be75811cf8),
        .name = make_view(device_name),
        .vendor = make_view(vendor_name),
        .reserved = {},
    };
    return BLACKFRAME_STATUS_SUCCESS;
}

auto BLACKFRAME_ABI_CALL get_interface(void*, const blackframe_interface_id interface_id,
                                       const uint32_t requested_major, const uint32_t,
                                       void* const out_interface,
                                       const uint32_t out_interface_size) noexcept
    -> blackframe_status {
    if (out_interface == nullptr || out_interface_size < sizeof(blackframe_xpu_backend_api_v1)) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }
    if (!is_xpu_interface(interface_id)) {
        return BLACKFRAME_STATUS_INTERFACE_NOT_FOUND;
    }
    if (requested_major != BLACKFRAME_XPU_BACKEND_ABI_MAJOR) {
        return BLACKFRAME_STATUS_UNSUPPORTED_ABI;
    }

    auto* const backend_api = static_cast<blackframe_xpu_backend_api_v1*>(out_interface);
    *backend_api = blackframe_xpu_backend_api_v1{
        .struct_size = sizeof(blackframe_xpu_backend_api_v1),
        .abi_major = BLACKFRAME_XPU_BACKEND_ABI_MAJOR,
        .abi_minor = BLACKFRAME_XPU_BACKEND_ABI_MINOR,
        .reserved = 0,
        .backend_context = nullptr,
        .backend_name = make_view(backend_name),
        .get_device_count = &get_device_count,
        .get_device_descriptor = &get_device_descriptor,
        .reserved_slots = {},
    };
    return BLACKFRAME_STATUS_SUCCESS;
}

void BLACKFRAME_ABI_CALL shutdown(void*) noexcept {}

} // namespace

extern "C" BLACKFRAME_ABI_EXPORT auto BLACKFRAME_ABI_CALL blackframe_query_extension(
    const blackframe_extension_query_v1* const query,
    blackframe_extension_api_v1* const out_extension) noexcept -> blackframe_status {
    if (query == nullptr || out_extension == nullptr ||
        query->struct_size < sizeof(blackframe_extension_query_v1) ||
        out_extension->struct_size < sizeof(blackframe_extension_api_v1)) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }
    if (query->abi_major != BLACKFRAME_EXTENSION_ABI_MAJOR) {
        return BLACKFRAME_STATUS_UNSUPPORTED_ABI;
    }

    *out_extension = blackframe_extension_api_v1{
        .struct_size = sizeof(blackframe_extension_api_v1),
        .abi_major = BLACKFRAME_EXTENSION_ABI_MAJOR,
        .abi_minor = BLACKFRAME_EXTENSION_ABI_MINOR,
        .reserved = 0,
        .extension_context = nullptr,
        .get_interface = &get_interface,
        .shutdown = &shutdown,
    };
    return BLACKFRAME_STATUS_SUCCESS;
}
