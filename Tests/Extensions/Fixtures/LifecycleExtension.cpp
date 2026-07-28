#include <Blackframe/Extensions/ExtensionAbi.h>
#include <cstdlib>

namespace {

struct LifecycleState {
    bool active{};
};

LifecycleState lifecycle_state;

struct UnloadGuard {
    ~UnloadGuard() {
        if (lifecycle_state.active) {
            std::abort();
        }
    }
};

UnloadGuard unload_guard;

blackframe_status BLACKFRAME_ABI_CALL get_interface(void* extension_context,
                                                    blackframe_interface_id, uint32_t, uint32_t,
                                                    void*, uint32_t) noexcept {
    if (extension_context != &lifecycle_state || !lifecycle_state.active) {
        return BLACKFRAME_STATUS_INVALID_STATE;
    }
    return BLACKFRAME_STATUS_INTERFACE_NOT_FOUND;
}

void BLACKFRAME_ABI_CALL shutdown(void* extension_context) noexcept {
    if (extension_context == &lifecycle_state) {
        lifecycle_state.active = false;
    }
}

} // namespace

extern "C" BLACKFRAME_ABI_EXPORT blackframe_status BLACKFRAME_ABI_CALL
blackframe_query_extension(const blackframe_extension_query_v1* query,
                           blackframe_extension_api_v1* out_extension) noexcept {
    if (query == nullptr || out_extension == nullptr ||
        query->struct_size < sizeof(blackframe_extension_query_v1) ||
        out_extension->struct_size < sizeof(blackframe_extension_api_v1) ||
        query->abi_major != BLACKFRAME_EXTENSION_ABI_MAJOR ||
        query->abi_minor > BLACKFRAME_EXTENSION_ABI_MINOR || query->reserved != 0) {
        return BLACKFRAME_STATUS_INVALID_ARGUMENT;
    }
    if (lifecycle_state.active) {
        return BLACKFRAME_STATUS_INVALID_STATE;
    }

    lifecycle_state.active = true;
    *out_extension = blackframe_extension_api_v1{
        .struct_size = sizeof(blackframe_extension_api_v1),
        .abi_major = BLACKFRAME_EXTENSION_ABI_MAJOR,
        .abi_minor = BLACKFRAME_EXTENSION_ABI_MINOR,
        .reserved = 0,
        .extension_context = &lifecycle_state,
        .get_interface = &get_interface,
        .shutdown = &shutdown,
    };
    return BLACKFRAME_STATUS_SUCCESS;
}
