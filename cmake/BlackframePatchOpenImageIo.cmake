cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED OPENIMAGEIO_SOURCE_DIR OR
   NOT EXISTS "${OPENIMAGEIO_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "OPENIMAGEIO_SOURCE_DIR must name the populated OpenImageIO source.")
endif()

set(openimageio_cmake_path "${OPENIMAGEIO_SOURCE_DIR}/CMakeLists.txt")
file(READ "${openimageio_cmake_path}" openimageio_cmake)
string(FIND
    "${openimageio_cmake}"
    "\"\${CMAKE_SOURCE_DIR}/src/include\""
    source_include_position
)
if(NOT source_include_position EQUAL -1)
    string(
        REPLACE
        "\"\${CMAKE_SOURCE_DIR}/src/include\""
        "\"\${PROJECT_SOURCE_DIR}/src/include\""
        openimageio_cmake
        "${openimageio_cmake}"
    )
    file(WRITE "${openimageio_cmake_path}" "${openimageio_cmake}")
elseif(NOT openimageio_cmake MATCHES "PROJECT_SOURCE_DIR}/src/include")
    message(FATAL_ERROR "The pinned OpenImageIO source include entry was not found.")
endif()

set(blackframe_patch_marker "# Blackframe: package exports belong to top-level OIIO builds only.")
string(FIND "${openimageio_cmake}" "${blackframe_patch_marker}" patch_position)
if(patch_position EQUAL -1)
    set(export_block_start
        "# Export the configuration files. There are also library-specific config\n"
    )
    string(FIND "${openimageio_cmake}" "${export_block_start}" export_start)
    if(export_start EQUAL -1)
        message(FATAL_ERROR "The pinned OpenImageIO export block start was not found.")
    endif()

    set(export_block_end
        "        NAMESPACE \${PROJECT_NAME}::)\n"
    )
    string(FIND "${openimageio_cmake}" "${export_block_end}" export_end)
    if(export_end EQUAL -1)
        message(FATAL_ERROR "The pinned OpenImageIO export block end was not found.")
    endif()
    string(LENGTH "${export_block_end}" export_block_end_length)
    math(EXPR export_end "${export_end} + ${export_block_end_length}")

    string(SUBSTRING "${openimageio_cmake}" 0 ${export_start} prefix)
    string(SUBSTRING "${openimageio_cmake}" ${export_start} -1 export_and_suffix)
    math(EXPR export_block_length "${export_end} - ${export_start}")
    string(SUBSTRING "${export_and_suffix}" 0 ${export_block_length} export_block)
    string(SUBSTRING "${openimageio_cmake}" ${export_end} -1 suffix)

    set(patched_cmake
        "${prefix}${blackframe_patch_marker}\nif (PROJECT_IS_TOP_LEVEL)\n${export_block}endif ()\n${suffix}"
    )
    file(WRITE "${openimageio_cmake_path}" "${patched_cmake}")
endif()

# A static TIFF carrying an imported libdeflate target cannot be re-exported
# reliably from OIIO's nested dependency build. ZIP/Deflate TIFF compression
# remains available through the already-required zlib dependency.
set(openimageio_tiff_recipe
    "${OPENIMAGEIO_SOURCE_DIR}/src/cmake/build_TIFF.cmake"
)
file(READ "${openimageio_tiff_recipe}" tiff_recipe)
string(FIND "${tiff_recipe}" "-D libdeflate=OFF" tiff_patch_position)
if(tiff_patch_position EQUAL -1)
    string(REPLACE "-D libdeflate=ON" "-D libdeflate=OFF" patched_tiff_recipe "${tiff_recipe}")
    if(patched_tiff_recipe STREQUAL tiff_recipe)
        message(FATAL_ERROR "The pinned OpenImageIO TIFF libdeflate option was not found.")
    endif()
    file(WRITE "${openimageio_tiff_recipe}" "${patched_tiff_recipe}")
endif()

# libdeflate 1.23 assumes target-feature implications that Clang 22 no longer
# provides on Windows. Disable only the affected optional AVX-512 paths; the
# portable and remaining runtime-dispatched implementations stay available.
set(openimageio_libdeflate_recipe
    "${OPENIMAGEIO_SOURCE_DIR}/src/cmake/build_libdeflate.cmake"
)
file(READ "${openimageio_libdeflate_recipe}" libdeflate_recipe)
set(
    libdeflate_patch_marker
    "# Blackframe: avoid unsupported Clang 22 AVX-512 feature implications."
)
string(FIND "${libdeflate_recipe}" "${libdeflate_patch_marker}" libdeflate_patch_position)
if(libdeflate_patch_position EQUAL -1)
    set(libdeflate_version_identifier [=[string (MAKE_C_IDENTIFIER ${libdeflate_BUILD_VERSION} libdeflate_VERSION_IDENT)
]=])
    string(CONCAT patched_libdeflate_version_identifier
        "${libdeflate_version_identifier}\n"
        "${libdeflate_patch_marker}\n"
        "set (_libdeflate_blackframe_clang_args)\n"
        "if (WIN32 AND CMAKE_C_COMPILER_ID STREQUAL \"Clang\" AND\n"
        "    CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 22)\n"
        "    list (APPEND _libdeflate_blackframe_clang_args\n"
        "          \"-DCMAKE_C_FLAGS:STRING=-DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_VPCLMULQDQ -DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX512VNNI\")\n"
        "endif ()\n"
    )
    string(
        REPLACE
        "${libdeflate_version_identifier}"
        "${patched_libdeflate_version_identifier}"
        patched_libdeflate_recipe
        "${libdeflate_recipe}"
    )
    if(patched_libdeflate_recipe STREQUAL libdeflate_recipe)
        message(FATAL_ERROR "The pinned OpenImageIO libdeflate setup was not found.")
    endif()

    set(libdeflate_cmake_args [=[    CMAKE_ARGS
]=])
    set(patched_libdeflate_cmake_args [=[    CMAKE_ARGS
        ${_libdeflate_blackframe_clang_args}
]=])
    string(
        REPLACE
        "${libdeflate_cmake_args}"
        "${patched_libdeflate_cmake_args}"
        patched_libdeflate_recipe_with_args
        "${patched_libdeflate_recipe}"
    )
    if(patched_libdeflate_recipe_with_args STREQUAL patched_libdeflate_recipe)
        message(FATAL_ERROR "The pinned OpenImageIO libdeflate arguments were not found.")
    endif()
    file(WRITE "${openimageio_libdeflate_recipe}" "${patched_libdeflate_recipe_with_args}")
endif()

# OpenColorIO 2.5.1's optional fast-math SIMD layer calls intrinsics that are
# unavailable with Clang 22's Windows GNU frontend. Its supported scalar path
# preserves functionality and avoids patching third-party implementation code.
set(openimageio_opencolorio_recipe
    "${OPENIMAGEIO_SOURCE_DIR}/src/cmake/build_OpenColorIO.cmake"
)
file(READ "${openimageio_opencolorio_recipe}" opencolorio_recipe)
set(
    opencolorio_patch_marker
    "# Blackframe: use OpenColorIO's scalar path with Clang 22 on Windows."
)
string(FIND "${opencolorio_recipe}" "${opencolorio_patch_marker}" opencolorio_patch_position)
if(opencolorio_patch_position EQUAL -1)
    set(opencolorio_version_identifier [=[string (MAKE_C_IDENTIFIER ${OpenColorIO_BUILD_VERSION} OpenColorIO_VERSION_IDENT)
]=])
    string(CONCAT patched_opencolorio_version_identifier
        "${opencolorio_version_identifier}\n"
        "${opencolorio_patch_marker}\n"
        "set (_opencolorio_blackframe_clang_args)\n"
        "if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL \"Clang\" AND\n"
        "    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)\n"
        "    list (APPEND _opencolorio_blackframe_clang_args\n"
        "          -D OCIO_USE_SIMD=OFF)\n"
        "endif ()\n"
    )
    string(
        REPLACE
        "${opencolorio_version_identifier}"
        "${patched_opencolorio_version_identifier}"
        patched_opencolorio_recipe
        "${opencolorio_recipe}"
    )
    if(patched_opencolorio_recipe STREQUAL opencolorio_recipe)
        message(FATAL_ERROR "The pinned OpenImageIO OpenColorIO setup was not found.")
    endif()

    set(opencolorio_cmake_args [=[    CMAKE_ARGS
]=])
    set(patched_opencolorio_cmake_args [=[    CMAKE_ARGS
        ${_opencolorio_blackframe_clang_args}
]=])
    string(
        REPLACE
        "${opencolorio_cmake_args}"
        "${patched_opencolorio_cmake_args}"
        patched_opencolorio_recipe_with_args
        "${patched_opencolorio_recipe}"
    )
    if(patched_opencolorio_recipe_with_args STREQUAL patched_opencolorio_recipe)
        message(FATAL_ERROR "The pinned OpenImageIO OpenColorIO arguments were not found.")
    endif()
    file(
        WRITE
        "${openimageio_opencolorio_recipe}"
        "${patched_opencolorio_recipe_with_args}"
    )
endif()

# OIIO configures its locally built dependencies in separate CMake processes.
# Disable user-wide vcpkg integration in Visual Studio projects so those
# builds cannot mix headers from an unrelated installation with the pinned
# static libraries installed in OIIO's private dependency prefix.
set(openimageio_dependency_utils
    "${OPENIMAGEIO_SOURCE_DIR}/src/cmake/dependency_utils.cmake"
)
file(READ "${openimageio_dependency_utils}" dependency_utils)
set(vcpkg_patch_marker "# Blackframe: isolate nested builds from user-wide vcpkg integration.")
string(FIND "${dependency_utils}" "${vcpkg_patch_marker}" vcpkg_patch_position)
if(vcpkg_patch_position EQUAL -1)
    string(CONCAT runtime_forwarding_block
        "    if (WIN32 AND CMAKE_MSVC_RUNTIME_LIBRARY)\n"
        "        list (APPEND _pkg_CMAKE_ARGS -DCMAKE_MSVC_RUNTIME_LIBRARY=\${CMAKE_MSVC_RUNTIME_LIBRARY})\n"
        "    endif ()\n"
    )
    string(CONCAT vcpkg_isolation_block
        "${runtime_forwarding_block}\n"
        "    ${vcpkg_patch_marker}\n"
        "    if (WIN32)\n"
        "        list (APPEND _pkg_CMAKE_ARGS\n"
        "              -DCMAKE_VS_GLOBALS=VcpkgEnabled=false\n"
        "              -DZLIB_USE_STATIC_LIBS=ON)\n"
        "    endif ()\n"
    )
    string(
        REPLACE
        "${runtime_forwarding_block}"
        "${vcpkg_isolation_block}"
        patched_dependency_utils
        "${dependency_utils}"
    )
    if(patched_dependency_utils STREQUAL dependency_utils)
        message(FATAL_ERROR "The pinned OpenImageIO dependency forwarding block was not found.")
    endif()
    file(WRITE "${openimageio_dependency_utils}" "${patched_dependency_utils}")
endif()

# OIIO's helper otherwise lets each nested CMake invocation select the host
# default generator and compilers, and it ignores failed configure/build/install
# processes. Keep every transitive build on the parent toolchain and fail before
# a stale or partial install can be accepted.
file(READ "${openimageio_dependency_utils}" dependency_utils)
set(toolchain_patch_marker "# Blackframe: inherit and verify the parent build toolchain.")
string(FIND "${dependency_utils}" "${toolchain_patch_marker}" toolchain_patch_position)
if(toolchain_patch_position EQUAL -1)
    set(nested_build_directory [=[    set (${pkgname}_LOCAL_BUILD_DIR "${${PROJECT_NAME}_LOCAL_DEPS_ROOT}/${pkgname}-build")
]=])
    set(patched_nested_build_directory [=[    # Blackframe: isolate build trees by the inherited toolchain.
    string (MAKE_C_IDENTIFIER
            "${CMAKE_GENERATOR}-${CMAKE_GENERATOR_PLATFORM}-${CMAKE_GENERATOR_TOOLSET}-${CMAKE_C_COMPILER_ID}-${CMAKE_CXX_COMPILER_ID}-${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}"
            _pkg_build_identity)
    set (${pkgname}_LOCAL_BUILD_DIR
         "${${PROJECT_NAME}_LOCAL_DEPS_ROOT}/${pkgname}-build-${_pkg_build_identity}")
]=])
    string(
        REPLACE
        "${nested_build_directory}"
        "${patched_nested_build_directory}"
        patched_dependency_utils
        "${dependency_utils}"
    )
    if(patched_dependency_utils STREQUAL dependency_utils)
        message(FATAL_ERROR "The pinned OpenImageIO nested build directory was not found.")
    endif()
    set(dependency_utils_with_build_directory "${patched_dependency_utils}")

    set(nested_process_block [=[    execute_process (COMMAND
        ${CMAKE_COMMAND}
            # Put things in our special local build areas
                -S ${${pkgname}_LOCAL_SOURCE_DIR}/${_pkg_SOURCE_SUBDIR}
                -B ${${pkgname}_LOCAL_BUILD_DIR}
                -DCMAKE_INSTALL_PREFIX=${${pkgname}_LOCAL_INSTALL_DIR}
            # Same build type as us
                -DCMAKE_BUILD_TYPE=${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            # Shhhh
                -DCMAKE_MESSAGE_INDENT="        "
                -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
                ${_pkg_cmake_verbose}
            # Build args passed by caller
                ${_pkg_CMAKE_ARGS}
        ${_pkg_exec_quiet}
        )

    # Build the package
    execute_process (COMMAND ${CMAKE_COMMAND}
                        --build ${${pkgname}_LOCAL_BUILD_DIR}
                        --config ${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                     ${_pkg_exec_quiet}
                    )

    # Install the project, unless instructed not to do so
    if (NOT _pkg_NOINSTALL)
        execute_process (COMMAND ${CMAKE_COMMAND}
                            --build ${${pkgname}_LOCAL_BUILD_DIR}
                            --config ${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                            --target install
                         ${_pkg_exec_quiet}
                        )
        set (${pkgname}_ROOT ${${pkgname}_LOCAL_INSTALL_DIR})
        list (APPEND CMAKE_PREFIX_PATH ${${pkgname}_LOCAL_INSTALL_DIR})
    endif ()
]=])
    set(patched_nested_process_block [=[    # Blackframe: inherit and verify the parent build toolchain.
    set (_pkg_GENERATOR_ARGS -G "${CMAKE_GENERATOR}")
    if (CMAKE_GENERATOR_PLATFORM)
        list (APPEND _pkg_GENERATOR_ARGS -A "${CMAKE_GENERATOR_PLATFORM}")
    endif ()
    if (CMAKE_GENERATOR_TOOLSET)
        list (APPEND _pkg_GENERATOR_ARGS -T "${CMAKE_GENERATOR_TOOLSET}")
    endif ()
    if (CMAKE_GENERATOR_INSTANCE)
        list (APPEND _pkg_CMAKE_ARGS
              "-DCMAKE_GENERATOR_INSTANCE=${CMAKE_GENERATOR_INSTANCE}")
    endif ()
    foreach (_pkg_forwarded_variable IN ITEMS
             CMAKE_MAKE_PROGRAM
             CMAKE_C_COMPILER
             CMAKE_CXX_COMPILER
             CMAKE_C_COMPILER_TARGET
             CMAKE_CXX_COMPILER_TARGET
             CMAKE_SYSROOT
             CMAKE_OSX_ARCHITECTURES
             CMAKE_OSX_DEPLOYMENT_TARGET)
        if (NOT "${${_pkg_forwarded_variable}}" STREQUAL "")
            list (APPEND _pkg_CMAKE_ARGS
                  "-D${_pkg_forwarded_variable}=${${_pkg_forwarded_variable}}")
        endif ()
    endforeach ()

    execute_process (COMMAND
        ${CMAKE_COMMAND}
                ${_pkg_GENERATOR_ARGS}
            # Put things in our special local build areas
                -S ${${pkgname}_LOCAL_SOURCE_DIR}/${_pkg_SOURCE_SUBDIR}
                -B ${${pkgname}_LOCAL_BUILD_DIR}
                -DCMAKE_INSTALL_PREFIX=${${pkgname}_LOCAL_INSTALL_DIR}
            # Same build type as us
                -DCMAKE_BUILD_TYPE=${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            # Shhhh
                -DCMAKE_MESSAGE_INDENT="        "
                -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
                ${_pkg_cmake_verbose}
            # Build args passed by caller
                ${_pkg_CMAKE_ARGS}
        ${_pkg_exec_quiet}
        RESULT_VARIABLE _pkg_configure_result
        )
    if (NOT "${_pkg_configure_result}" STREQUAL "0")
        message (FATAL_ERROR
                 "Local ${pkgname} configure failed with ${_pkg_configure_result}.")
    endif ()

    # Build the package
    execute_process (COMMAND ${CMAKE_COMMAND}
                        --build ${${pkgname}_LOCAL_BUILD_DIR}
                        --config ${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                     ${_pkg_exec_quiet}
                     RESULT_VARIABLE _pkg_build_result
                    )
    if (NOT "${_pkg_build_result}" STREQUAL "0")
        message (FATAL_ERROR
                 "Local ${pkgname} build failed with ${_pkg_build_result}.")
    endif ()

    # Install the project, unless instructed not to do so
    if (NOT _pkg_NOINSTALL)
        execute_process (COMMAND ${CMAKE_COMMAND}
                            --build ${${pkgname}_LOCAL_BUILD_DIR}
                            --config ${${PROJECT_NAME}_DEPENDENCY_BUILD_TYPE}
                            --target install
                         ${_pkg_exec_quiet}
                         RESULT_VARIABLE _pkg_install_result
                        )
        if (NOT "${_pkg_install_result}" STREQUAL "0")
            message (FATAL_ERROR
                     "Local ${pkgname} install failed with ${_pkg_install_result}.")
        endif ()
        set (${pkgname}_ROOT ${${pkgname}_LOCAL_INSTALL_DIR})
        list (APPEND CMAKE_PREFIX_PATH ${${pkgname}_LOCAL_INSTALL_DIR})
    endif ()
]=])
    string(
        REPLACE
        "${nested_process_block}"
        "${patched_nested_process_block}"
        patched_dependency_utils
        "${dependency_utils_with_build_directory}"
    )
    if(patched_dependency_utils STREQUAL dependency_utils_with_build_directory)
        message(FATAL_ERROR "The pinned OpenImageIO nested process block was not found.")
    endif()
    file(WRITE "${openimageio_dependency_utils}" "${patched_dependency_utils}")
endif()

# The generated toolchain includes any caller-provided toolchain. Do not append
# the original path again after it, which could override the compiler lock for
# presets that select compilers outside their base toolchain file.
file(READ "${openimageio_dependency_utils}" dependency_utils)
set(toolchain_loop_with_parent [=[             CMAKE_MAKE_PROGRAM
             CMAKE_TOOLCHAIN_FILE
             CMAKE_C_COMPILER
]=])
set(toolchain_loop_without_parent [=[             CMAKE_MAKE_PROGRAM
             CMAKE_C_COMPILER
]=])
string(FIND "${dependency_utils}" "${toolchain_loop_with_parent}" parent_toolchain_position)
if(NOT parent_toolchain_position EQUAL -1)
    string(
        REPLACE
        "${toolchain_loop_with_parent}"
        "${toolchain_loop_without_parent}"
        patched_dependency_utils
        "${dependency_utils}"
    )
    file(WRITE "${openimageio_dependency_utils}" "${patched_dependency_utils}")
elseif(NOT dependency_utils MATCHES "CMAKE_MAKE_PROGRAM[\r\n ]+CMAKE_C_COMPILER")
    message(FATAL_ERROR "The pinned OpenImageIO forwarded-variable list was not found.")
endif()

# This marker is independent of the surrounding toolchain patch so a hot
# source tree produced by an older Blackframe patch gains transitive forwarding
# on the next configure instead of silently retaining the old behavior.
file(READ "${openimageio_dependency_utils}" dependency_utils)
set(
    transitive_toolchain_patch_marker
    "# Blackframe: forward the generated transitive toolchain."
)
string(
    FIND
    "${dependency_utils}"
    "${transitive_toolchain_patch_marker}"
    transitive_toolchain_patch_position
)
if(transitive_toolchain_patch_position EQUAL -1)
    set(generator_instance_forwarding [=[    if (CMAKE_GENERATOR_INSTANCE)
        list (APPEND _pkg_CMAKE_ARGS
              "-DCMAKE_GENERATOR_INSTANCE=${CMAKE_GENERATOR_INSTANCE}")
    endif ()
]=])
    string(CONCAT transitive_toolchain_forwarding
        "${generator_instance_forwarding}"
        "    ${transitive_toolchain_patch_marker}\n"
        "    if (NOT \"\${\${PROJECT_NAME}_DEPENDENCY_TOOLCHAIN_FILE}\" STREQUAL \"\")\n"
        "        list (APPEND _pkg_CMAKE_ARGS\n"
        "              \"-DCMAKE_TOOLCHAIN_FILE=\${\${PROJECT_NAME}_DEPENDENCY_TOOLCHAIN_FILE}\")\n"
        "    endif ()\n"
    )
    string(
        REPLACE
        "${generator_instance_forwarding}"
        "${transitive_toolchain_forwarding}"
        patched_dependency_utils
        "${dependency_utils}"
    )
    if(patched_dependency_utils STREQUAL dependency_utils)
        message(FATAL_ERROR "The pinned generator-instance forwarding block was not found.")
    endif()
    file(WRITE "${openimageio_dependency_utils}" "${patched_dependency_utils}")
endif()

# Embedded readers are the complete runtime format set. Do not scan process
# environment paths for additional plugin binaries, which would make accepted
# formats and executed code depend on the launching machine.
set(openimageio_plugin_source
    "${OPENIMAGEIO_SOURCE_DIR}/src/libOpenImageIO/imageioplugin.cpp"
)
file(READ "${openimageio_plugin_source}" plugin_source)
set(plugin_patch_marker "// Blackframe: embedded image readers are the closed plugin set.")
string(FIND "${plugin_source}" "${plugin_patch_marker}" plugin_patch_position)
if(plugin_patch_position EQUAL -1)
    set(plugin_scan_start
        "    append_if_env_exists(searchpath, \"OPENIMAGEIO_PLUGIN_PATH\", true);\n"
    )
    string(CONCAT patched_plugin_scan_start
        "    ${plugin_patch_marker}\n"
        "#ifndef EMBED_PLUGINS\n"
        "${plugin_scan_start}"
    )
    string(
        REPLACE
        "${plugin_scan_start}"
        "${patched_plugin_scan_start}"
        patched_plugin_source
        "${plugin_source}"
    )
    if(patched_plugin_source STREQUAL plugin_source)
        message(FATAL_ERROR "The pinned OpenImageIO external plugin scan start was not found.")
    endif()
    set(plugin_source_with_guard_start "${patched_plugin_source}")

    set(plugin_scan_end
        "    }\n\n    // Inventory the procedural plugins\n"
    )
    string(CONCAT patched_plugin_scan_end
        "    }\n"
        "#else\n"
        "    (void)searchpath;\n"
        "#endif\n\n"
        "    // Inventory the procedural plugins\n"
    )
    string(
        REPLACE
        "${plugin_scan_end}"
        "${patched_plugin_scan_end}"
        patched_plugin_source
        "${plugin_source_with_guard_start}"
    )
    if(patched_plugin_source STREQUAL plugin_source_with_guard_start)
        message(FATAL_ERROR "The pinned OpenImageIO external plugin scan end was not found.")
    endif()
    file(WRITE "${openimageio_plugin_source}" "${patched_plugin_source}")
endif()
