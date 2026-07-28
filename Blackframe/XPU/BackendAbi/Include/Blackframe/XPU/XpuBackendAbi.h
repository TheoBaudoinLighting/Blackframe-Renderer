#ifndef BLACKFRAME_XPU_XPU_BACKEND_ABI_H
#define BLACKFRAME_XPU_XPU_BACKEND_ABI_H

#include <Blackframe/Extensions/ExtensionAbi.h>
#include <stdint.h>

#define BLACKFRAME_XPU_BACKEND_ABI_MAJOR UINT32_C(1)
#define BLACKFRAME_XPU_BACKEND_ABI_MINOR UINT32_C(0)

#define BLACKFRAME_XPU_BACKEND_INTERFACE_ID_HIGH UINT64_C(0x96c6e70dd06b4b93)
#define BLACKFRAME_XPU_BACKEND_INTERFACE_ID_LOW UINT64_C(0xa9514ab777f49d37)

typedef uint32_t blackframe_xpu_device_kind;

#define BLACKFRAME_XPU_DEVICE_KIND_UNKNOWN ((blackframe_xpu_device_kind)0u)
#define BLACKFRAME_XPU_DEVICE_KIND_CPU ((blackframe_xpu_device_kind)1u)
#define BLACKFRAME_XPU_DEVICE_KIND_GPU ((blackframe_xpu_device_kind)2u)
#define BLACKFRAME_XPU_DEVICE_KIND_ACCELERATOR ((blackframe_xpu_device_kind)3u)

/*
 * Names are owned by the backend and remain valid until extension shutdown.
 * Device identifiers must be stable and unique within one backend.
 */
typedef struct blackframe_xpu_device_descriptor_v1 {
    uint32_t struct_size;
    blackframe_xpu_device_kind kind;
    uint64_t identifier_high;
    uint64_t identifier_low;
    blackframe_utf8_view name;
    blackframe_utf8_view vendor;
    uint64_t reserved[4];
} blackframe_xpu_device_descriptor_v1;

typedef blackframe_status(BLACKFRAME_ABI_CALL* blackframe_xpu_get_device_count_fn)(
    void* backend_context, uint32_t* out_device_count) BLACKFRAME_ABI_NOEXCEPT;

typedef blackframe_status(BLACKFRAME_ABI_CALL* blackframe_xpu_get_device_descriptor_fn)(
    void* backend_context, uint32_t device_index,
    blackframe_xpu_device_descriptor_v1* out_descriptor) BLACKFRAME_ABI_NOEXCEPT;

/*
 * XPU ABI v1 deliberately exposes enumeration only. Rendering, memory,
 * queues, kernels, and synchronization will each require an explicit design.
 */
typedef struct blackframe_xpu_backend_api_v1 {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t reserved;
    void* backend_context;
    blackframe_utf8_view backend_name;
    blackframe_xpu_get_device_count_fn get_device_count;
    blackframe_xpu_get_device_descriptor_fn get_device_descriptor;
    uint64_t reserved_slots[4];
} blackframe_xpu_backend_api_v1;

#endif
