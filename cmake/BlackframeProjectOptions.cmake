include_guard(GLOBAL)

function(blackframe_create_project_options)
    add_library(BlackframeProjectOptions INTERFACE)
    add_library(Blackframe::ProjectOptions ALIAS BlackframeProjectOptions)
    blackframe_set_target_role(BlackframeProjectOptions core)

    target_compile_features(BlackframeProjectOptions INTERFACE cxx_std_26)

    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        )
    else()
        target_compile_options(
            BlackframeProjectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
                $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
                $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
        )
    endif()

    if(BLACKFRAME_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CXX>:/WX>
            )
        else()
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
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
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND
           NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "BLACKFRAME_ENABLE_SANITIZERS requires Clang or GCC.")
        elseif(WIN32)
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
            )
            target_link_options(BlackframeProjectOptions INTERFACE -fsanitize=address)
        else()
            target_compile_options(
                BlackframeProjectOptions
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
                    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
            )
            target_link_options(
                BlackframeProjectOptions
                INTERFACE
                    -fsanitize=address,undefined
            )
        endif()
    endif()
endfunction()
