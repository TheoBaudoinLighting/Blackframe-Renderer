cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED TEST_EXECUTABLE OR NOT EXISTS "${TEST_EXECUTABLE}")
    message(FATAL_ERROR "The PNG checksum test executable does not exist: ${TEST_EXECUTABLE}")
endif()
if(NOT DEFINED OUTPUT_PATH OR NOT IS_ABSOLUTE "${OUTPUT_PATH}")
    message(FATAL_ERROR "The PNG checksum output path must be absolute.")
endif()
string(LENGTH "${EXPECTED_SHA256}" expected_sha256_length)
if(NOT DEFINED EXPECTED_SHA256 OR
   NOT expected_sha256_length EQUAL 64 OR
   NOT EXPECTED_SHA256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "The expected PNG SHA-256 checksum is malformed.")
endif()

file(REMOVE "${OUTPUT_PATH}")
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "BLACKFRAME_PNG_CHECKSUM_OUTPUT=${OUTPUT_PATH}"
        "${TEST_EXECUTABLE}"
        "--gtest_filter=PngWriterTest.WritesFixedDisplayTransformForSyntheticCrop"
        "--gtest_color=no"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_output
    ERROR_VARIABLE test_error
)
if(NOT test_result EQUAL 0)
    message(
        FATAL_ERROR
        "The synthetic PNG generation failed.\n${test_output}\n${test_error}"
    )
endif()
if(NOT EXISTS "${OUTPUT_PATH}")
    message(FATAL_ERROR "The synthetic PNG was not written: ${OUTPUT_PATH}")
endif()

file(SHA256 "${OUTPUT_PATH}" actual_sha256)
file(REMOVE "${OUTPUT_PATH}")
if(NOT actual_sha256 STREQUAL EXPECTED_SHA256)
    message(
        FATAL_ERROR
        "Synthetic PNG SHA-256 is '${actual_sha256}', expected '${EXPECTED_SHA256}'."
    )
endif()
