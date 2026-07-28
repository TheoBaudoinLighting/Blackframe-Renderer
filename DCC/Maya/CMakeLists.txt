if(NOT DEFINED MAYA_LOCATION)
    message(
        FATAL_ERROR
        "BLACKFRAME_BUILD_DCC_MAYA=ON requires MAYA_LOCATION to point to a Maya SDK installation."
    )
endif()

find_path(
    BLACKFRAME_MAYA_INCLUDE_DIRECTORY
    maya/MFnPlugin.h
    HINTS "${MAYA_LOCATION}"
    PATH_SUFFIXES include
    REQUIRED
)

find_library(
    BLACKFRAME_MAYA_FOUNDATION_LIBRARY
    Foundation
    HINTS "${MAYA_LOCATION}"
    PATH_SUFFIXES lib lib64
    REQUIRED
)

find_library(
    BLACKFRAME_MAYA_OPENMAYA_LIBRARY
    OpenMaya
    HINTS "${MAYA_LOCATION}"
    PATH_SUFFIXES lib lib64
    REQUIRED
)

add_library(BlackframeMayaPlugin MODULE Source/MayaPluginEntry.cpp)
blackframe_set_target_role(BlackframeMayaPlugin tool)

target_include_directories(
    BlackframeMayaPlugin
    SYSTEM PRIVATE
        "${BLACKFRAME_MAYA_INCLUDE_DIRECTORY}"
)

target_link_libraries(
    BlackframeMayaPlugin
    PRIVATE
        Blackframe::DccSdk
        "${BLACKFRAME_MAYA_FOUNDATION_LIBRARY}"
        "${BLACKFRAME_MAYA_OPENMAYA_LIBRARY}"
)

set_target_properties(
    BlackframeMayaPlugin
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME Blackframe
)

if(WIN32)
    set_target_properties(BlackframeMayaPlugin PROPERTIES SUFFIX ".mll")
endif()
