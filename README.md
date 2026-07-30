# Blackframe

Blackframe is a small headless rendering-engine foundation written in strict C++26. It currently
provides deterministic, testable contracts for renderer math, geometry, sampling and color, film
storage, image output, validation, local IPC, and native backend discovery.

It is not yet a complete path tracer. The implemented surface is deliberately narrow. Backend
availability is reported explicitly, and requesting an unavailable backend fails instead of
silently selecting another path.

## What exists today

- **Precision and math:** `float` transport and distinct `double` reference values, checked
  conversions, separate vectors/points/normals, matrices, quaternions, affine and projection
  transforms, robust inverses, local frames, intervals, bounds, and AABB tests.
- **Rays and camera:** bounded rays carrying time, visibility mask, and current-medium identity;
  scale-aware origin offsets; pinhole primary rays; deterministic or jittered subpixel sampling.
- **Geometry:** analytic sphere, plane, and disk intersections plus watertight triangles. Primitive
  hit records expose the relevant roots, UVs, normals, or barycentrics.
- **Surface data:** a separate validated `SurfaceInteraction` contract stores position, geometric
  and shading normals, UVs, derivatives, identifiers, and time.
- **Internal scene:** the Engine exposes an immutable frame snapshot storing a closed object,
  geometry, material, and nested-instance graph under explicit stable 32-bit identifiers.
  Instances carry validated local affine matrices and optional parents; world transforms are
  resolved deterministically when the snapshot closes. Duplicate IDs, dangling references, cycles,
  invalid compositions, and lookup failures are explicit errors.
- **Triangle meshes:** the Engine imports strictly triangulated OBJ text and PLY ASCII 1.0 assets
  from explicit absolute paths. Positions, unit normals, UVs, and 32-bit indices are validated;
  independent OBJ index domains are canonicalized at seams. Missing attributes, polygons, invalid
  indices, degenerate triangles, unsupported PLY encodings, and malformed records fail explicitly.
  A deterministic compaction pass removes unreferenced and bit-identical aligned vertices while
  preserving distinct seams. Its byte accounting derives from the four contiguous POD buffers that
  a CPU renderer or future CUDA upload consumes, with an expanded-corner comparison and no allocator
  or container-capacity estimate.
- **Transport state:** validated float and double `PathState` values own spectral throughput,
  accumulated radiance, bound category depth counters, refraction scaling, wavelengths, delta
  history, and medium identity, with a versioned bit-exact diagnostic dump. A bounded scalar
  BSDF-only loop linearly traces explicit wavelength-resolved diffuse triangle surfaces,
  accumulates every encountered emitter or environment, and spawns continuation rays from derived
  position-error bounds without NEE, MIS, or a hidden backend. Explicit depth budgets count
  diffuse, glossy, specular, transmission, and volume events separately; transmitted surface
  events advance both their scattering family and transmission counters. The current Lambertian
  loop consumes only diffuse-reflection depth and supports explicitly configured, compensated
  Russian roulette from a selected completed depth.
- **Sampling:** indexed `SampleStream` values with a versioned dimension map, independent hashing,
  local PCG32, stratification, Latin hypercube, high-dimensional Sobol, reproducible Owen
  scrambling, and common disk/sphere/hemisphere mappings. No global mutable RNG is used.
- **Spectral and color foundations:** four-lane 360-830 nm wavelength packets with marginal PDFs,
  black, constant, and tabulated spectra, `SampledSpectrum<4>`, energy-conserving spectral
  Lambertian reflection, one-sided spectral surface emission, a constant spectral environment, the
  CIE 1931 2-degree observer to relative XYZ, and signed scene-linear RGB.
- **Film and output:** weighted float accumulation, compensated double reference accumulation,
  crops, deterministic tile fusion, a tested scene-linear 32-bit RGB OpenEXR writer, and an
  optional stb-backed PNG preview writer with a fixed display transform.
- **Validation:** linear and HDR error metrics, display-referred PSNR and heatmaps, plus debug
  encodings for normals, depth, UVs, barycentrics, and identifiers covered by five 64x64 goldens.
  Two closed CornellDiffuse validation fixtures provide tracked 64x64 and 256x256 scene-linear EXR
  references rendered by `scalar_ref` at 4096 and 1024 spp. Their scenes, generator source snapshot,
  and image hashes are verified before deterministic 1-versus-4-spp MSE, RMSE, and display-PSNR
  convergence checks.
- **Host control:** bounded versioned local IPC, a C extension ABI, explicit absolute-path loading,
  XPU device discovery, a reference discovery plugin, and a headless `render` control executable.
- **Backend integration:** explicit capability reporting and pre-dispatch checks, a pinned Embree 4
  smoke test, and an optional CUDA C++20 smoke kernel. These are integration contracts, not complete
  rendering backends.

## Current boundary

Blackframe does not yet provide a production path-tracing integrator, a scene loader, a general
material or lighting system, a production acceleration pipeline, or a CUDA wavefront renderer. The
current BSDF-only loop is a narrow deterministic scalar oracle over resolved Lambertian triangles,
not a production backend. The headless executable currently supports local engine control and
device discovery; it does not yet accept a scene and render an image from the command line. A
consumable CMake install/export package is also not available yet. The CornellDiffuse JSON files
are closed validation descriptors, not a general scene-interchange format or a hidden scene loader.

## Building

Requirements:

- CMake 3.30 or newer.
- Ninja.
- A host compiler with strict C++26 support, including pack indexing. The supplied presets use
  `clang` and `clang++`.
- On Windows, compatible Visual Studio C++ build tools and a Windows SDK for the MSVC-targeting
  Clang toolchain. Windows CUDA presets must run from a Visual Studio developer environment.
- An internet connection for the first configuration, unless the pinned FetchContent archives are
  already populated below the selected build directory.
- CUDA Toolkit 13.3.33 only when a CUDA preset is selected. Linux expects `nvcc` in `PATH`; CUDA
  architectures must be explicit, and the supplied presets target architecture 86.

OpenEXR, Imath, Embree, stb, GoogleTest, and Google Benchmark are fetched at immutable revisions
with verified hashes as required by the selected configuration. All dependency sources and build
artifacts remain below `build/`; no manual third-party source checkout is required. Enabling a
dependency-backed feature without its required toolchain is a configuration error. Each
configuration also writes `blackframe-dependencies.json` and its SHA-256 file in the preset build
directory.

Windows CPU debug:

```powershell
cmake --preset windows-cpu-debug
cmake --build --preset windows-cpu-debug
ctest --preset windows-cpu-debug
```

Linux CPU debug:

```sh
cmake --preset linux-cpu-debug
cmake --build --preset linux-cpu-debug
ctest --preset linux-cpu-debug
```

The corresponding `*-cpu-release` presets enable benchmarks. CPU sanitizer and CUDA presets are
also declared for both platforms in `CMakePresets.json`. The main switches are `BUILD_TESTING` and
the `BLACKFRAME_BUILD_*` / `BLACKFRAME_ENABLE_*` options declared in the root `CMakeLists.txt`;
enabled but unavailable dependencies fail configuration.

## Headless control service

After a Windows debug build:

```powershell
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe --capabilities
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe --help
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe serve
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe request ping
```

Run `serve` and `request` in separate terminals. XPU plugins are accepted only by `serve` and must
be supplied through an explicit absolute path.
