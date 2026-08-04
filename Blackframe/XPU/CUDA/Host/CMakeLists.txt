add_library(
    BlackframeCudaMemory
    STATIC
        Source/DeviceMemory.cpp
)
add_library(Blackframe::CudaMemory ALIAS BlackframeCudaMemory)
blackframe_set_target_role(BlackframeCudaMemory cuda)

target_sources(
    BlackframeCudaMemory
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/DeviceMemory.hpp"
)

target_include_directories(
    BlackframeCudaMemory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:Include>
)

target_link_libraries(
    BlackframeCudaMemory
    PUBLIC
        Blackframe::Core
    PRIVATE
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
)
