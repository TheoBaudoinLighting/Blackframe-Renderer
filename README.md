# Blackframe

Blackframe is a headless spectral rendering engine written in strict C++26 on the host and CUDA
C++20 on the device. It currently renders deterministic four-wavelength scenes through a readable
scalar reference, an Embree CPU wavefront backend, and a CUDA wavefront backend.

The engine is built as a numerical and architectural foundation rather than a general-purpose
production renderer. Backends and optional capabilities are selected explicitly: an unavailable
or unsupported path fails before rendering instead of silently choosing a substitute.

## Current capabilities

### Rendering core

- Transport uses `float`; reference calculations and optional compensated accumulation use
  `double`. Conversions, ABI sizes, offsets, and host/device alignments are tested.
- Vectors, points, and normals are distinct types. The math layer includes matrices, quaternions,
  affine and projection transforms, robust inverses, local frames, intervals, bounds, and AABB
  intersection.
- Rays carry a bounded parameter interval, time, visibility mask, and current medium identifier.
  Primary pinhole rays support deterministic center sampling or indexed subpixel jitter.
- Scalar paths propagate full ray differentials across ideal specular events. CPU and CUDA
  wavefront paths carry explicit ray cones, advance their world-space footprint at hits, and apply
  the same reflection, Snell, diffuse, and GGX spread rules without changing the traversal ABI.
- Analytic spheres, planes, and disks coexist with watertight triangles. Surface interactions keep
  geometric and shading normals separate and expose UVs, derivatives, stable IDs, and time.
- Immutable `FrameScene` snapshots own stable object, geometry, material, texture, light, and
  instance identifiers. Nested transforms are resolved when the snapshot closes.
- Strict OBJ and ASCII PLY import validates positions, normals, UVs, and triangle indices.
  Compacted meshes retain contiguous position, normal, UV, and index storage without duplicating
  aligned vertices unnecessarily.
- Scene updates distinguish full rebuilds from transform-only refits. Incompatible refits fail
  instead of rebuilding implicitly.

### Spectral transport and materials

- Each path carries four wavelengths in the 360–830 nm visible range together with their sampling
  PDFs. Constant, black, and tabulated spectra feed CIE 1931 XYZ and signed scene-linear RGB.
- `SampleStream` indexes every value by pixel, sample, dimension, and seed. Independent hashing,
  PCG32, stratified and Latin-hypercube sampling, high-dimensional Sobol with Owen scrambling, and
  common geometric mappings are deterministic and use no global RNG.
- `ClosureSet` stores at most eight closures inline, without allocating at a hit. Explicit
  probabilities form mixtures whose values and PDFs include component selection exactly once.
- Active closures include Lambert, energy-preserving Oren–Nayar, ideal mirror and specular
  transmission, GGX conductor and dielectric, and rotated anisotropic GGX.
- Fresnel dielectric and wavelength-dependent complex conductor Fresnel use the full equations.
  Delta and continuous events retain distinct probability measures.
- Scene transport supports spectral surface emission, a constant environment, punctual lights,
  and emissive meshes. Next-event estimation and balance or power MIS avoid double counting.
- Diffuse, glossy, specular, transmission, and volume depth budgets are tracked separately.
  Russian roulette is configurable and compensated.

### Execution backends

| Backend | Role | Acceleration and execution |
| --- | --- | --- |
| Scalar reference | Independent readable oracle | Analytic traversal and compensated reference film |
| CPU wavefront | Production CPU path | Embree 4, bounded queues, stable compaction, one or many workers |
| CUDA wavefront | Production GPU path | CUDA BLAS/TLAS, device queues, kernels, compaction, streams, and events |

All three paths share the same wavelength, PDF, event, closure, light, sampler, and path-state
conventions. The CUDA backend traverses its own deterministic BVH; it never uses Embree or hidden
CPU execution. Synchronous and asynchronous CUDA modes are selected when the workspace is created,
and mode mismatches are errors.

The CPU scheduler records queue capacity, occupancy, overflow, dispatches, lanes, and stage time.
CUDA adds bounded SoA queues, stable prefix-scan compaction, memory and transfer telemetry, optional
NVTX ranges, and reusable RAII workspaces with explicit out-of-memory reporting.

### Film, output, and images

- Film storage keeps weighted sums, weights, and sample counts per pixel. It supports float
  accumulation, compensated double reference accumulation, crops, tiles, and deterministic merges.
- Output includes scene-linear 32-bit RGB OpenEXR and an optional PNG preview with fixed exposure
  and display transform.
- OpenImageIO loads immutable interleaved-float host images and caches them by canonical path and
  mandatory color-space tag. PNM/PFM, PNG, and EXR are covered by tests.
- Data and scene-linear values stay unchanged. Tagged sRGB channels use a fixed IEC decode while
  alpha and extra channels remain untouched; tags are never guessed.
- Host sampling supports nearest, bilinear, and Catmull–Rom bicubic reconstruction with per-tap
  repeat, clamp, mirrored-repeat, or black-border addressing.
- Mip pyramids are generated from the converted snapshot down to 1x1 with exact-area box averages,
  including odd extents. Trilinear sampling blends bilinear results at an explicit finite LOD.
- EWA filtering consumes an explicit UV footprint and bounds anisotropy and texel visits. Host
  evaluation supports float transport and double reference precision; CUDA surface maps use the
  same bounded float contract.
- Spectral materials can bind data-tagged tangent-space normal maps and filtered bump maps by
  texture ID. Normal-map Y orientation is explicit, bump gradients preserve the current shading
  normal, and geometric normals remain unchanged for visibility and ray offsets. `scalar_ref`
  uses ray differentials; CPU and CUDA wavefront paths use propagated ray cones.
- Host UDIM descriptors resolve one `<UDIM>` token with standard ten-column numbering and return
  the addressed image plus tile-local UVs. Missing tiles fail explicitly instead of selecting a
  neighbor, tile 1001, or a diagnostic color.
- Constant float, color, and spectrum textures are evaluated by scalar, CPU, and CUDA backends.
  Invalid tags, modes, coordinates, channels, budgets, or LODs fail explicitly.

### Validation

GoogleTest and CTest cover algebra, transforms, geometry, sampling, PDFs, white-furnace energy,
scene lifetime, acceleration parity, queue behavior, image I/O, and deterministic replay. Google
Benchmark records performance and quality reports without putting fragile timing limits in unit
tests.

The same-sample scalar/CPU and scalar/CUDA transport matrices currently require:

| Metric | Threshold |
| --- | ---: |
| Linear MSE | at most `1e-10` |
| Linear RMSE | at most `1e-5` |
| Maximum absolute error | at most `1e-4` |
| Display PSNR | at least `80 dB`, or positive infinity |

The tracked CornellDiffuse controls include 64x64 at 4096 spp and 256x256 at 1024 spp scene-linear
references. The canonical 256-spp render has a neutral enclosure, red and green walls, a ceiling
emitter, and exactly two white Lambertian spheres. Validation also reports relative MSE, luminance
SMAPE, mean bias, heatmaps, time, rays, samples, queue statistics, and memory where applicable.

## Current limitations

- There is no general scene-file loader or command-line scene renderer yet. The Cornell JSON files
  are closed validation descriptors, not an interchange format.
- Transport is vacuum-only. Unsupported media are rejected.
- Image textures currently bind only to normal and bump material slots; reflectance and emission
  image parameters are not connected yet. Blackframe does not invent an implicit RGB-to-spectrum
  conversion.
- Ray cones are circular footprints. Their ideal reflection/transmission bound assumes a locally
  constant closure frame; interpolated-normal variation is not bounded. Continuous lobes have
  unbounded support, so diffuse and GGX use documented finite spread policies. Full anisotropic
  derivatives remain a `scalar_ref` validation capability rather than per-lane wavefront state.
- A latitude-longitude environment-map light contract exists, but scene transport currently
  resolves only the constant environment; maps are not connected to misses, NEE, or MIS.
- Geometry updates cover full rebuilds and frame-to-frame transform refits, not deformation or
  transform motion blur.
- The public MIS entry points start complete primary paths; `PathState` does not carry a prior
  vertex's directional PDFs.
- The headless executable currently exposes engine control and device discovery rather than a
  general render command.
- A consumable CMake install/export package is not available yet.

## Building

### Requirements

- CMake 3.30 or newer and Ninja.
- A compiler with strict C++26 host support, including pack indexing. Supplied CPU presets use
  Clang.
- On Windows, compatible Visual Studio C++ build tools and a Windows SDK. CUDA presets must run
  from a Visual Studio developer environment.
- CUDA Toolkit 13.3.33 for CUDA presets. The supplied presets target architecture 86 and CUDA tests
  use the matching `compute-sanitizer`.
- Internet access for the first configuration unless all pinned dependency archives are already
  present below the chosen build directory. OpenImageIO dependency setup also requires Git.

OpenImageIO, OpenEXR, Imath, Embree, stb, GoogleTest, and Google Benchmark are fetched at immutable
revisions with verified hashes as required by the selected preset. Dependency sources, generated
files, tests, reports, and render artifacts remain below `build/`; no manual third-party checkout is
required. Configuration fails when an enabled capability or toolchain is unavailable.

Every configuration writes `blackframe-dependencies.json` and its SHA-256 file in the selected
build directory. Important switches include `BUILD_TESTING`, `BLACKFRAME_BUILD_BENCHMARKS`,
`BLACKFRAME_ENABLE_EMBREE`, `BLACKFRAME_ENABLE_OPENIMAGEIO`, `BLACKFRAME_ENABLE_STB`, and
`BLACKFRAME_ENABLE_CUDA`.

### CPU presets

Windows debug:

```powershell
cmake --preset windows-cpu-debug
cmake --build --preset windows-cpu-debug
ctest --preset windows-cpu-debug
```

Linux debug:

```sh
cmake --preset linux-cpu-debug
cmake --build --preset linux-cpu-debug
ctest --preset linux-cpu-debug
```

The corresponding `*-cpu-release` presets enable benchmarks. Sanitizer presets are also declared
in [CMakePresets.json](CMakePresets.json).

### CUDA preset

From a Visual Studio developer shell on Windows:

```powershell
cmake --preset windows-cuda-debug
cmake --build --preset windows-cuda-debug
ctest --preset windows-cuda-debug
```

## Render and validation

Configure and build the CPU Release preset before running the CPU validation commands:

```powershell
cmake --preset windows-cpu-release
cmake --build --preset windows-cpu-release
```

The canonical Cornell image is produced by the validation benchmark, not by a hidden scene loader:

```powershell
ctest --test-dir build/windows-cpu-release `
  -R '^Blackframe\.Benchmarks\.CornellPowerMisConvergence\.Json$' `
  --output-on-failure
```

Run the complete scalar-to-CPU wavefront parity inventory with:

```powershell
ctest --test-dir build/windows-cpu-release `
  -R '^Blackframe\.EmbreeScalarWavefrontParity\.Json$' `
  --output-on-failure
```

The CUDA stable-compaction benchmark and its JSON contract use:

```powershell
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
ctest --test-dir build/windows-cuda-release `
  -R '^Blackframe\.Benchmarks\.CudaWavefrontCompaction\.Json$' `
  --output-on-failure
```

Nsight Systems validation is opt-in and requires an absolute executable path:

```powershell
cmake --preset windows-cuda-debug `
  -DBLACKFRAME_ENABLE_NSIGHT_VALIDATION=ON `
  -DBLACKFRAME_NSYS_EXECUTABLE='C:\absolute\path\to\nsys.exe'
ctest --test-dir build/windows-cuda-debug -C Debug `
  -R '^BlackframeCudaNsightValidation$' --output-on-failure
```

## Headless control service

After a Windows CPU debug build:

```powershell
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe --capabilities
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe --help
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe serve
.\build\windows-cpu-debug\Blackframe\Applications\Headless\render.exe request ping
```

Run `serve` and `request` in separate terminals. XPU plugins are accepted only by `serve` and must
be supplied through an explicit absolute path.

## Repository map

- [CMake presets](CMakePresets.json) and [build options](CMakeLists.txt)
- [Pinned dependencies](cmake/BlackframeDependencies.cmake)
- [Public renderer headers](Blackframe/Renderer/Include/Blackframe/Renderer/)
- [Engine](Blackframe/Engine/)
- [Embree backend](Blackframe/Backends/CPU/Embree/)
- [CUDA backend](Blackframe/Backends/GPU/CUDA/)
- [Tests](Tests/) and [benchmarks](Benchmarks/)
- [Cornell scenes](scenes/) and
  [reference manifest](scenes/references/S02_CornellDiffuse.references.json)
