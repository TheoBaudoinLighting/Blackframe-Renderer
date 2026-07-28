include_guard(GLOBAL)

function(_blackframe_prepare_windows_asan_runtime)
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine
        RESULT_VARIABLE compiler_target_result
        OUTPUT_VARIABLE compiler_target
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE compiler_target_error
    )
    if(NOT compiler_target_result EQUAL 0)
        message(
            FATAL_ERROR
            "Cannot determine the Clang target needed by AddressSanitizer:\n"
            "${compiler_target_error}"
        )
    endif()

    if(compiler_target MATCHES "^(x86_64|amd64)")
        set(runtime_architecture x86_64)
    elseif(compiler_target MATCHES "^(aarch64|arm64)")
        set(runtime_architecture aarch64)
    else()
        message(
            FATAL_ERROR
            "Windows AddressSanitizer is not configured for compiler target "
            "'${compiler_target}'."
        )
    endif()

    set(runtime_name "clang_rt.asan_dynamic-${runtime_architecture}.dll")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" "--print-file-name=${runtime_name}"
        RESULT_VARIABLE runtime_query_result
        OUTPUT_VARIABLE runtime_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE runtime_query_error
    )
    cmake_path(CONVERT "${runtime_path}" TO_CMAKE_PATH_LIST runtime_path NORMALIZE)
    if(NOT runtime_query_result EQUAL 0 OR
       runtime_path STREQUAL runtime_name OR
       NOT EXISTS "${runtime_path}")
        message(
            FATAL_ERROR
            "Cannot locate the Windows AddressSanitizer runtime '${runtime_name}':\n"
            "${runtime_query_error}"
        )
    endif()

    set(runtime_directory "${CMAKE_BINARY_DIR}/sanitizer-runtime")
    file(MAKE_DIRECTORY "${runtime_directory}")
    file(
        COPY_FILE
        "${runtime_path}"
        "${runtime_directory}/${runtime_name}"
        ONLY_IF_DIFFERENT
    )
    set(
        BLACKFRAME_SANITIZER_RUNTIME_DIRECTORY
        "${runtime_directory}"
        CACHE INTERNAL
        "Directory containing the runtime required by sanitizer test presets"
        FORCE
    )
    message(STATUS "Blackframe sanitizer runtime: ${runtime_path}")
endfunction()

function(_blackframe_instrument_target target visibility)
    if(WIN32)
        target_compile_options(
            "${target}"
            "${visibility}"
                $<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>
                $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>
        )
        target_compile_definitions(
            "${target}"
            "${visibility}"
                _DISABLE_STRING_ANNOTATION
                _DISABLE_VECTOR_ANNOTATION
        )
    else()
        target_compile_options(
            "${target}"
            "${visibility}"
                $<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>
                $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address,undefined>
                $<$<COMPILE_LANGUAGE:C,CXX>:-fno-sanitize-recover=all>
        )
    endif()
endfunction()

function(blackframe_enable_dependency_sanitizers target)
    if(NOT BLACKFRAME_ENABLE_SANITIZERS)
        return()
    endif()
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Cannot sanitize unknown dependency target '${target}'.")
    endif()

    _blackframe_instrument_target("${target}" PRIVATE)
endfunction()

function(blackframe_create_project_options)
    add_library(BlackframeProjectOptions INTERFACE)
    add_library(Blackframe::ProjectOptions ALIAS BlackframeProjectOptions)
    blackframe_set_target_role(BlackframeProjectOptions core)

    target_compile_features(BlackframeProjectOptions INTERFACE cxx_std_26)

    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
        )
    else()
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wsign-conversion>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wshadow>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wformat=2>
                $<$<COMPILE_LANGUAGE:C,CXX>:-Wundef>
                $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
                $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
        )
    endif()

    if(BLACKFRAME_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:C,CXX>:/WX>
            )
        else()
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
            )
        endif()
    endif()

    if(WIN32)
        target_compile_definitions(
            BlackframeProjectOptions
            INTERFACE
                NOMINMAX
                WIN32_LEAN_AND_MEAN
                UNICODE
                _UNICODE
        )
    endif()

    if(BLACKFRAME_ENABLE_SANITIZERS)
        if(BLACKFRAME_ENABLE_CUDA)
            message(FATAL_ERROR "Sanitizers are supported only by CPU presets.")
        endif()
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|GNU)$")
            message(FATAL_ERROR "BLACKFRAME_ENABLE_SANITIZERS requires Clang or GCC.")
        elseif(WIN32)
            _blackframe_instrument_target(BlackframeProjectOptions INTERFACE)
            target_link_options(BlackframeProjectOptions INTERFACE -fsanitize=address)
            _blackframe_prepare_windows_asan_runtime()
        else()
            _blackframe_instrument_target(BlackframeProjectOptions INTERFACE)
            target_link_options(
                BlackframeProjectOptions
                INTERFACE
                    -fsanitize=address,undefined
                    -fno-sanitize-recover=all
            )
        endif()
    endif()
endfunction()
