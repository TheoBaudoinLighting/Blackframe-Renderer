cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED REPOSITORY_SOURCE_DIR OR NOT IS_ABSOLUTE "${REPOSITORY_SOURCE_DIR}")
    message(FATAL_ERROR "REPOSITORY_SOURCE_DIR must be an explicit absolute path.")
endif()

set(reference_index
    "${REPOSITORY_SOURCE_DIR}/scenes/references/S02_CornellDiffuse.references.json"
)
if(NOT EXISTS "${reference_index}")
    message(FATAL_ERROR "The Cornell reference index is missing: ${reference_index}")
endif()

file(READ "${reference_index}" reference_manifest)
string(JSON manifest_schema GET "${reference_manifest}" schema)
string(JSON manifest_version GET "${reference_manifest}" schema_version)
string(JSON source_base_commit GET "${reference_manifest}" source_snapshot base_commit)
string(JSON generator_source_count LENGTH "${reference_manifest}" source_snapshot files)
string(JSON reference_count LENGTH "${reference_manifest}" references)
string(LENGTH "${source_base_commit}" source_base_commit_length)
if(NOT manifest_schema STREQUAL "blackframe.cornell_reference_set")
    message(FATAL_ERROR "The Cornell reference index uses an unsupported schema.")
endif()
if(NOT manifest_version EQUAL 1)
    message(FATAL_ERROR "The Cornell reference index uses an unsupported version.")
endif()
if(NOT source_base_commit_length EQUAL 40 OR NOT source_base_commit MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "The Cornell reference source base is not a full lowercase commit hash.")
endif()
if(NOT generator_source_count EQUAL 6)
    message(FATAL_ERROR "The Cornell reference source snapshot must contain exactly six files.")
endif()
if(NOT reference_count EQUAL 2)
    message(FATAL_ERROR "The Cornell reference index must contain exactly two entries.")
endif()

set(
    expected_generator_sources
    "Tests/Renderer/CMakeLists.txt"
    "Tests/Renderer/CornellDiffuseImageRenderer.hpp"
    "Tests/Renderer/CornellDiffuseReferenceGenerator.cpp"
    "Tests/Renderer/CornellDiffuseReferenceImage.hpp"
    "Tests/Renderer/CornellDiffuseReferenceTest.cpp"
    "Tests/Renderer/CornellDiffuseTestScene.hpp"
)
foreach(source_index RANGE 0 5)
    list(GET expected_generator_sources "${source_index}" expected_source)
    string(
        JSON actual_source
        GET "${reference_manifest}" source_snapshot files "${source_index}" path
    )
    string(
        JSON declared_source_hash
        GET "${reference_manifest}" source_snapshot files "${source_index}" sha256
    )
    string(LENGTH "${declared_source_hash}" declared_source_hash_length)
    if(NOT actual_source STREQUAL expected_source OR
       NOT declared_source_hash_length EQUAL 64 OR
       NOT declared_source_hash MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Cornell generator source entry ${source_index} is invalid.")
    endif()

    set(source_path "${REPOSITORY_SOURCE_DIR}/${actual_source}")
    if(NOT EXISTS "${source_path}")
        message(FATAL_ERROR "Cornell generator source entry ${source_index} is missing.")
    endif()
    file(SHA256 "${source_path}" actual_source_hash)
    if(NOT actual_source_hash STREQUAL declared_source_hash)
        message(FATAL_ERROR "Cornell generator source entry ${source_index} changed.")
    endif()
endforeach()

set(expected_variants 64x64 256x256)
set(expected_widths 64 256)
set(expected_heights 64 256)
set(expected_spp 4096 1024)
set(
    expected_scenes
    "scenes/S02_CornellDiffuse_64x64.json"
    "scenes/S02_CornellDiffuse_256x256.json"
)
set(
    expected_images
    "scenes/references/S02_CornellDiffuse_64x64_scalar_ref_4096spp.exr"
    "scenes/references/S02_CornellDiffuse_256x256_scalar_ref_1024spp.exr"
)

foreach(reference_index RANGE 0 1)
    list(GET expected_variants "${reference_index}" expected_variant)
    list(GET expected_widths "${reference_index}" expected_width)
    list(GET expected_heights "${reference_index}" expected_height)
    list(GET expected_spp "${reference_index}" expected_sample_count)
    list(GET expected_scenes "${reference_index}" expected_scene)
    list(GET expected_images "${reference_index}" expected_image)

    string(
        JSON actual_variant
        GET "${reference_manifest}" references "${reference_index}" variant
    )
    string(
        JSON actual_scene
        GET "${reference_manifest}" references "${reference_index}" scene
    )
    string(
        JSON declared_scene_hash
        GET "${reference_manifest}" references "${reference_index}" scene_sha256
    )
    string(
        JSON actual_image
        GET "${reference_manifest}" references "${reference_index}" image
    )
    string(
        JSON declared_image_hash
        GET "${reference_manifest}" references "${reference_index}" image_sha256
    )
    string(
        JSON actual_sample_count
        GET "${reference_manifest}" references "${reference_index}" samples_per_pixel
    )
    string(
        JSON actual_seed
        GET "${reference_manifest}" references "${reference_index}" seed
    )
    string(LENGTH "${declared_scene_hash}" declared_scene_hash_length)
    string(LENGTH "${declared_image_hash}" declared_image_hash_length)

    if(NOT actual_variant STREQUAL expected_variant OR
       NOT actual_scene STREQUAL expected_scene OR
       NOT actual_image STREQUAL expected_image OR
       NOT actual_sample_count EQUAL expected_sample_count OR
       NOT actual_seed STREQUAL "0x243F6A8885A308D3")
        message(FATAL_ERROR "Cornell reference entry ${reference_index} is incompatible.")
    endif()
    if(NOT declared_scene_hash_length EQUAL 64 OR
       NOT declared_image_hash_length EQUAL 64 OR
       NOT declared_scene_hash MATCHES "^[0-9a-f]+$" OR
       NOT declared_image_hash MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Cornell reference entry ${reference_index} has an invalid SHA-256.")
    endif()

    set(scene_path "${REPOSITORY_SOURCE_DIR}/${actual_scene}")
    set(image_path "${REPOSITORY_SOURCE_DIR}/${actual_image}")
    if(NOT EXISTS "${scene_path}" OR NOT EXISTS "${image_path}")
        message(FATAL_ERROR "Cornell reference entry ${reference_index} names a missing asset.")
    endif()
    file(SHA256 "${scene_path}" actual_scene_hash)
    file(SHA256 "${image_path}" actual_image_hash)
    if(NOT actual_scene_hash STREQUAL declared_scene_hash OR
       NOT actual_image_hash STREQUAL declared_image_hash)
        message(FATAL_ERROR "Cornell reference entry ${reference_index} failed SHA-256 validation.")
    endif()

    file(READ "${scene_path}" scene_manifest)
    string(JSON scene_schema GET "${scene_manifest}" schema)
    string(JSON scene_version GET "${scene_manifest}" schema_version)
    string(JSON scene_id GET "${scene_manifest}" scene_id)
    string(JSON scene_variant GET "${scene_manifest}" variant)
    string(JSON scene_width GET "${scene_manifest}" film width)
    string(JSON scene_height GET "${scene_manifest}" film height)
    string(JSON scene_reference_spp GET "${scene_manifest}" reference samples_per_pixel)
    string(JSON scene_reference_image GET "${scene_manifest}" reference image)
    string(JSON scene_reference_seed GET "${scene_manifest}" reference seed)
    if(NOT scene_schema STREQUAL "blackframe.cornell_diffuse_fixture" OR
       NOT scene_version EQUAL 1 OR
       NOT scene_id STREQUAL "S02_CornellDiffuse" OR
       NOT scene_variant STREQUAL expected_variant OR
       NOT scene_width EQUAL expected_width OR
       NOT scene_height EQUAL expected_height OR
       NOT scene_reference_spp EQUAL expected_sample_count OR
       NOT scene_reference_image STREQUAL expected_image OR
       NOT scene_reference_seed STREQUAL actual_seed)
        message(FATAL_ERROR "Cornell scene ${actual_scene} disagrees with its reference entry.")
    endif()
endforeach()

message(STATUS "Cornell source, scene, and image hashes are valid.")
