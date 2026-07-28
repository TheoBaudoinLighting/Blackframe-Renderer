#ifndef BLACKFRAME_EXTENSIONS_EXTENSION_ABI_H
#define BLACKFRAME_EXTENSIONS_EXTENSION_ABI_H

#include <stdint.h>

#define BLACKFRAME_EXTENSION_ABI_MAJOR UINT32_C(1)
#define BLACKFRAME_EXTENSION_ABI_MINOR UINT32_C(0)
#define BLACKFRAME_EXTENSION_QUERY_SYMBOL "blackframe_query_extension"

#if defined(__cplusplus)
#define BLACKFRAME_ABI_EXTERN_C extern "C"
#define BLACKFRAME_ABI_NOEXCEPT noexcept
#else
#define BLACKFRAME_ABI_EXTERN_C
#define BLACKFRAME_ABI_NOEXCEPT
#endif

#if defined(_WIN32)
#define BLACKFRAME_ABI_CALL __cdecl
#if defined(BLACKFRAME_EXTENSION_IMPLEMENTATION)
#define BLACKFRAME_ABI_EXPORT __declspec(dllexport)
#else
#define BLACKFRAME_ABI_EXPORT
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define BLACKFRAME_ABI_CALL
#if defined(BLACKFRAME_EXTENSION_IMPLEMENTATION)
#define BLACKFRAME_ABI_EXPORT __attribute__((visibility("default")))
#else
#define BLACKFRAME_ABI_EXPORT
#endif
#else
#define BLACKFRAME_ABI_CALL
#define BLACKFRAME_ABI_EXPORT
#endif

typedef uint32_t blackframe_status;

#define BLACKFRAME_STATUS_SUCCESS ((blackframe_status)0u)
#define BLACKFRAME_STATUS_INVALID_ARGUMENT ((blackframe_status)1u)
#define BLACKFRAME_STATUS_UNSUPPORTED_ABI ((blackframe_status)2u)
#define BLACKFRAME_STATUS_INTERFACE_NOT_FOUND ((blackframe_status)3u)
#define BLACKFRAME_STATUS_INVALID_STATE ((blackframe_status)4u)
#define BLACKFRAME_STATUS_OUT_OF_RANGE ((blackframe_status)5u)
#define BLACKFRAME_STATUS_EXTENSION_FAILURE ((blackframe_status)6u)

typedef struct blackframe_interface_id {
    uint64_t high;
    uint64_t low;
} blackframe_interface_id;

/*
 * The referenced bytes are UTF-8 and remain owned by the provider. Unless a
 * more specific interface states otherwise, they remain valid until shutdown.
 */
typedef struct blackframe_utf8_view {
    const char* data;
    uint32_t size;
    uint32_t reserved;
} blackframe_utf8_view;

typedef struct blackframe_extension_query_v1 {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t reserved;
} blackframe_extension_query_v1;

typedef blackframe_status(BLACKFRAME_ABI_CALL* blackframe_extension_get_interface_fn)(
    void* extension_context, blackframe_interface_id interface_id, uint32_t requested_major,
    uint32_t requested_minor, void* out_interface,
    uint32_t out_interface_size) BLACKFRAME_ABI_NOEXCEPT;

typedef void(BLACKFRAME_ABI_CALL* blackframe_extension_shutdown_fn)(void* extension_context)
    BLACKFRAME_ABI_NOEXCEPT;

/*
 * Before querying, the host sets struct_size to sizeof(this structure).
 * A successful extension replaces it with the size of the structure it wrote.
 */
typedef struct blackframe_extension_api_v1 {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t reserved;
    void* extension_context;
    blackframe_extension_get_interface_fn get_interface;
    blackframe_extension_shutdown_fn shutdown;
} blackframe_extension_api_v1;

typedef blackframe_status(BLACKFRAME_ABI_CALL* blackframe_query_extension_fn)(
    const blackframe_extension_query_v1* query,
    blackframe_extension_api_v1* out_extension) BLACKFRAME_ABI_NOEXCEPT;

/*
 * This is the sole public entry point required from a Blackframe extension.
 * No C++ type, exception, allocator, or ownership crosses this boundary.
 */
BLACKFRAME_ABI_EXTERN_C BLACKFRAME_ABI_EXPORT blackframe_status BLACKFRAME_ABI_CALL
blackframe_query_extension(const blackframe_extension_query_v1* query,
                           blackframe_extension_api_v1* out_extension) BLACKFRAME_ABI_NOEXCEPT;

#endif
