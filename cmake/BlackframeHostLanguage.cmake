include_guard(GLOBAL)

macro(blackframe_require_host_cxx26)
    list(FIND CMAKE_CXX_COMPILE_FEATURES "cxx_std_26" blackframe_cxx26_feature_index)
    if(blackframe_cxx26_feature_index EQUAL -1)
        message(
            FATAL_ERROR
            "Blackframe requires C++26 for every host target, but "
            "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' does not advertise "
            "the cxx_std_26 compile feature. Blackframe does not fall back to an older standard."
        )
    endif()
    unset(blackframe_cxx26_feature_index)

    try_compile(
        blackframe_cxx26_probe_compiled
        SOURCE_FROM_CONTENT
            BlackframeCxx26Probe.cpp
            [=[
#include <cstddef>

#if !defined(__cpp_pack_indexing) || __cpp_pack_indexing < 202311L
#error Blackframe requires C++26 pack indexing support.
#endif

template <std::size_t index, typename... Values>
consteval auto select_pack_value(Values... values) {
    return values...[index];
}

static_assert(select_pack_value<2>(11, 22, 33, 44) == 33);

int main() {
    return 0;
}
]=]
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        NO_CACHE
        OUTPUT_VARIABLE blackframe_cxx26_probe_output
    )
    if(NOT blackframe_cxx26_probe_compiled)
        message(
            FATAL_ERROR
            "Blackframe requires strict C++26 host support, including pack indexing. "
            "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' failed the probe. "
            "Blackframe does not fall back to an older standard.\n"
            "${blackframe_cxx26_probe_output}"
        )
    endif()
    unset(blackframe_cxx26_probe_compiled)
    unset(blackframe_cxx26_probe_output)

    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endmacro()
