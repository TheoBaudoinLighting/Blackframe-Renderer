#include <Blackframe/Extensions/ExtensionAbi.h>
#include <cstdlib>

namespace {

bool extension_active = false;

struct UnloadGuard {
    ~UnloadGuard() {
        if (extension_active) {
            std::abort();
        }
    }
};

UnloadGuard unload_guard;

blackframe_status BLACKFRAME_ABI_CALL get_interface(void*, blackframe_interface_id, uint32_t,
                                                    uint32_t, void*, uint32_t) noexcept {
    return BLACKFRAME_STATUS_INTERFACE_NOT_FOUND;
}

void BLACKFRAME_ABI_CALL shutdown(void*) noexcept {
    extension_active = false;
}

} // namespace

extern "C" BLACKFRAME_ABI_EXPORT blackframe_status BLACKFRAME_ABI_CALL
blackframe_query_extension(const blackframe_extension_query_v1* query,
                           blackframe_extension_api_v1* out_extension) noexcept {
    if (query == nullptr || out_extension == nullptr ||
        out_extension->struct_size < sizeof(blackframe_extension_api_v1)) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }

    extension_active = true;
    *out_extension = blackframe_extension_api_v1{
        .struct_size = sizeof(blackframe_extension_api_v1),
        .abi_major = BLACKFRAME_EXTENSION_ABI_MAJOR + 1,
        .abi_minor = 0,
        .reserved = 0,
        .extension_context = nullptr,
        .get_interface = &get_interface,
        .shutdown = &shutdown,
    };
    return BLACKFRAME_STATUS_SUCCESS;
}
