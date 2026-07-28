#include <Blackframe/Extensions/ExtensionAbi.h>
#include <stddef.h>

_Static_assert(sizeof(blackframe_interface_id) == 16, "Interface identifiers are 128-bit.");
_Static_assert(offsetof(blackframe_extension_api_v1, struct_size) == 0,
               "Every extension table starts with its byte size.");

int main(void) {
    return BLACKFRAME_EXTENSION_ABI_MAJOR == UINT32_C(1) ? 0 : 1;
}
