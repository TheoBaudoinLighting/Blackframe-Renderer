add_library(BlackframeXpuReferenceCpu MODULE)

target_sources(
    BlackframeXpuReferenceCpu
    PRIVATE
        Source/ReferenceCpuExtension.cpp
)

target_link_libraries(
    BlackframeXpuReferenceCpu
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::ExtensionAbi
        Blackframe::XpuBackendAbi
)

target_compile_definitions(
    BlackframeXpuReferenceCpu
    PRIVATE
        BLACKFRAME_EXTENSION_IMPLEMENTATION=1
)

set_target_properties(
    BlackframeXpuReferenceCpu
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        LIBRARY_OUTPUT_DIRECTORY "${blackframe_plugin_output_directory}/XPU"
        RUNTIME_OUTPUT_DIRECTORY "${blackframe_plugin_output_directory}/XPU"
        PREFIX ""
        OUTPUT_NAME "BlackframeXpuReferenceCpu"
)
