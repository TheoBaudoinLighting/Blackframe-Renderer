include_guard(GLOBAL)

function(blackframe_require_disposable_build_tree)
    file(REAL_PATH "${CMAKE_SOURCE_DIR}" blackframe_source_directory)
    file(REAL_PATH "${CMAKE_SOURCE_DIR}/build" blackframe_build_root)
    file(REAL_PATH "${CMAKE_BINARY_DIR}" blackframe_binary_directory)

    if(blackframe_source_directory STREQUAL blackframe_binary_directory)
        message(
            FATAL_ERROR
            "Blackframe does not support in-source builds. "
            "Remove generated CMake files from the source tree and run `cmake --preset dev`."
        )
    endif()

    cmake_path(
        IS_PREFIX
        blackframe_build_root
        "${blackframe_binary_directory}"
        NORMALIZE
        blackframe_binary_is_disposable
    )

    if(NOT blackframe_binary_is_disposable)
        message(
            FATAL_ERROR
            "All Blackframe build artifacts must live below '${blackframe_build_root}'. "
            "Configure with a preset or use `cmake -S . -B build/<name>`."
        )
    endif()
endfunction()
