# Blackframe

Blackframe is currently a small headless C++26 rendering-engine foundation. It provides typed
geometry and transforms, rays and a pinhole camera, deterministic film accumulation, OpenEXR and
PNG output, image-quality metrics, local IPC, native extension loading, and an Embree smoke path.

The repository builds exclusively with CMake and validates its current contracts with GoogleTest
and CTest. It does not yet contain a production path-tracing integrator or scene pipeline.
