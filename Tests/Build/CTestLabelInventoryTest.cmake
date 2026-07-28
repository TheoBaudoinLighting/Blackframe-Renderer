foreach(required_variable IN ITEMS CTEST_COMMAND BINARY_DIRECTORY REQUIRED_LABELS)
    if(NOT DEFINED "${required_variable}" OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

string(REPLACE "," ";" required_labels "${REQUIRED_LABELS}")
execute_process(
    COMMAND
        "${CTEST_COMMAND}"
        --test-dir
        "${BINARY_DIRECTORY}"
        --show-only=json-v1
    RESULT_VARIABLE inventory_result
    OUTPUT_VARIABLE inventory_output
    ERROR_VARIABLE inventory_error
)
if(NOT inventory_result EQUAL 0)
    message(
        FATAL_ERROR
        "CTest could not report its label inventory:\n"
        "${inventory_output}${inventory_error}"
    )
endif()

set(discovered_labels "")
string(JSON test_count LENGTH "${inventory_output}" tests)
if(test_count EQUAL 0)
    message(FATAL_ERROR "CTest did not discover any tests.")
endif()
math(EXPR last_test_index "${test_count} - 1")
foreach(test_index RANGE "${last_test_index}")
    string(
        JSON
        property_count
        LENGTH
        "${inventory_output}"
        tests
        "${test_index}"
        properties
    )
    if(property_count EQUAL 0)
        continue()
    endif()

    math(EXPR last_property_index "${property_count} - 1")
    foreach(property_index RANGE "${last_property_index}")
        string(
            JSON
            property_name
            GET
            "${inventory_output}"
            tests
            "${test_index}"
            properties
            "${property_index}"
            name
        )
        if(NOT property_name STREQUAL "LABELS")
            continue()
        endif()

        string(
            JSON
            label_count
            LENGTH
            "${inventory_output}"
            tests
            "${test_index}"
            properties
            "${property_index}"
            value
        )
        math(EXPR last_label_index "${label_count} - 1")
        foreach(label_index RANGE "${last_label_index}")
            string(
                JSON
                discovered_label
                GET
                "${inventory_output}"
                tests
                "${test_index}"
                properties
                "${property_index}"
                value
                "${label_index}"
            )
            list(APPEND discovered_labels "${discovered_label}")
        endforeach()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES discovered_labels)

foreach(label IN LISTS required_labels)
    if(NOT label IN_LIST discovered_labels)
        message(FATAL_ERROR "CTest label '${label}' is not discoverable.")
    endif()
endforeach()
