add_library(BlackframeDccSdk INTERFACE)
add_library(Blackframe::DccSdk ALIAS BlackframeDccSdk)
blackframe_set_target_role(BlackframeDccSdk tool)

target_sources(
    BlackframeDccSdk
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/DCC/SceneConventions.hpp>"
)

target_include_directories(
    BlackframeDccSdk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(BlackframeDccSdk INTERFACE Blackframe::ProjectOptions)
