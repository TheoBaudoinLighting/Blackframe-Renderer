include_guard(GLOBAL)

function(blackframe_create_project_options)
    add_library(BlackframeProjectOptions INTERFACE)
    add_library(Blackframe::ProjectOptions ALIAS BlackframeProjectOptions)

    target_compile_features(BlackframeProjectOptions INTERFACE cxx_std_26)

    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                /W4
                /permissive-
                /Zc:__cplusplus
                /EHsc
                /utf-8
        )
    else()
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wshadow
        )
    endif()

    if(BLACKFRAME_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_compile_options(BlackframeProjectOptions INTERFACE /WX)
        else()
            target_compile_options(BlackframeProjectOptions INTERFACE -Werror)
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
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND
           NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "BLACKFRAME_ENABLE_SANITIZERS requires Clang or GCC.")
        elseif(WIN32)
            target_compile_options(BlackframeProjectOptions INTERFACE -fsanitize=address)
            target_link_options(BlackframeProjectOptions INTERFACE -fsanitize=address)
        else()
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    -fno-omit-frame-pointer
                    -fsanitize=address,undefined
            )
            target_link_options(
                BlackframeProjectOptions
                INTERFACE
                    -fsanitize=address,undefined
            )
        endif()
    endif()
endfunction()
