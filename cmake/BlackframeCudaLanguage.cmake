include_guard(GLOBAL)

function(blackframe_require_explicit_cuda_architectures)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
       "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        message(
            FATAL_ERROR
            "BLACKFRAME_ENABLE_CUDA requires an explicit CMAKE_CUDA_ARCHITECTURES list."
        )
    endif()

    foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(NOT architecture MATCHES "^[0-9]+[a-z]?(-(real|virtual))?$")
            message(
                FATAL_ERROR
                "Unsupported CUDA architecture `${architecture}`. "
                "Use explicit architecture numbers such as `86` or `90-real`."
            )
        endif()
    endforeach()
endfunction()

function(blackframe_require_cuda_cxx20)
    if(NOT "cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
        message(
            FATAL_ERROR
            "The configured CUDA compiler does not support required CUDA C++20."
        )
    endif()
endfunction()

function(blackframe_configure_cuda_target target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Unknown CUDA target `${target}`.")
    endif()

    set_target_properties(
        "${target}"
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_features("${target}" PRIVATE cuda_std_20)

    if(WIN32)
        target_compile_options(
            "${target}"
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
        )
        if(BLACKFRAME_WARNINGS_AS_ERRORS)
            target_compile_options(
                "${target}"
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
                    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror all-warnings>
            )
        endif()
    else()
        target_compile_options(
            "${target}"
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
        if(BLACKFRAME_WARNINGS_AS_ERRORS)
            target_compile_options(
                "${target}"
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
                    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror all-warnings>
            )
        endif()
    endif()
endfunction()
