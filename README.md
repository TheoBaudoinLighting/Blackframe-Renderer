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
  Geometries retain immutable compact triangle meshes; instances carry validated local affine
  matrices, visibility masks, and optional parents. World transforms are resolved deterministically
  when the snapshot closes. A snapshot may remain geometry-only, or carry one explicit four-lane
  environment plus wavelength-resolved Lambertian reflection and one-sided emission for every
  material. The same packet also closes an insertion-ordered registry of point, directional, and
  spot lights whose slots remain stable for light sampling. Every non-black emissive mesh instance
  additionally derives a one-sided area-light model from that exact mesh, resolved world transform,
  and material emission, in stable instance-ID order; there is no second emitter description.
  Partially populated spectral scenes are rejected. Missing meshes, duplicate IDs, dangling
  references, cycles, invalid compositions, invalid lights, unrepresentable emissive transforms,
  and lookup failures are explicit errors.
- **Triangle meshes:** the Engine imports strictly triangulated OBJ text and PLY ASCII 1.0 assets
  from explicit absolute paths. Positions, unit normals, UVs, and 32-bit indices are validated;
  independent OBJ index domains are canonicalized at seams. Missing attributes, polygons, invalid
  indices, degenerate triangles, unsupported PLY encodings, and malformed records fail explicitly.
  A deterministic compaction pass removes unreferenced and bit-identical aligned vertices while
  preserving distinct seams. Its byte accounting derives from the four contiguous POD buffers that
  a CPU renderer or future CUDA upload consumes, with an expanded-corner comparison and no allocator
  or container-capacity estimate.
- **Acceleration queries:** one Engine-owned `AccelBackend` contract exposes closest-hit and
  occlusion queries over immutable frame scenes, stable object/surface identifiers, and visibility
  masks. The deterministic analytic oracle transforms rays into object space. The pinned Embree
  implementation shares one triangle BLAS per geometry and registers each logical instance in a
  TLAS using its resolved world transform; logical parent chains are flattened only after
  deterministic hierarchy resolution. The factory commits the initial snapshot; explicit rebuild
  and transform-only refit operations publish later immutable snapshots and report lifecycle
  counters. Embree refits the dynamic TLAS while retaining its triangle BLAS data. Incompatible
  refits fail instead of rebuilding implicitly. Shadow rays use direct Embree any-hit traversal over
  that TLAS, with 32-bit ray and per-instance visibility masks. Both implementations are selected
  through distinct factories returning the same interface and expose the exact committed snapshot
  used by scene queries. Backend failures or unsupported Embree requirements are reported without
  analytic substitution.
- **Transport state:** validated float and double `PathState` values own spectral throughput,
  accumulated radiance, bound category depth counters, refraction scaling, wavelengths, delta
  history, and medium identity, with a versioned bit-exact diagnostic dump. A bounded scalar
  BSDF-only loop linearly traces explicit wavelength-resolved diffuse triangle surfaces,
  accumulates every encountered emitter or environment, and spawns continuation rays from derived
  position-error bounds. The same bounce kernel can instead resolve every hit from the immutable
  frame scene through an explicitly supplied `AccelBackend`; positions, geometric and shading
  normals, UV derivatives, stable IDs, spectral materials, and emissions are reconstructed before
  continuation. Primary and secondary queries have no linear fallback on this path. A separate
  scene path samples exactly one registered punctual light at every accepted Lambertian vertex,
  evaluates its discrete light-selection and conditional probabilities once, and traces the
  resulting shadow ray through the selected acceleration backend. It rejects incomplete sampler
  support and non-vacuum visibility instead of changing estimators silently. A separate MIS path
  samples the combined punctual and emissive-mesh registry. Punctual delta samples retain weight
  one; continuous samples compare the joint `P(slot) * p_li` solid-angle density with the
  Lambertian PDF using an explicitly selected balance or power heuristic. The complementary weight
  is applied before accumulating an emissive surface reached by a BSDF continuation, so the two
  techniques do not double count the same transport path. Unsupported heuristic values, resumed
  non-delta paths missing their previous directional PDF, and incomplete light-sampler support fail
  explicitly. Explicit depth budgets count diffuse, glossy, specular, transmission, and volume
  events separately; transmitted surface events advance both their scattering family and
  transmission counters. The current Lambertian loops consume only diffuse-reflection depth and
  support explicitly configured, compensated Russian roulette from a selected completed depth.
- **CPU wavefront transport:** a versioned path-state structure of arrays and seven distinct bounded
  camera, ray, hit, miss, shade, shadow, and continuation queues provide fixed-capacity storage,
  guarded double buffering, and stable terminated-lane compaction with optional canonical path-slot
  ordering. A synchronous scheduler dispatches immutable queue snapshots over an explicit bounded
  worker count: one worker runs inline, while a persistent multi-worker pool receives deterministic
  contiguous lane partitions and reaches a barrier before queue publication. The current Embree
  transport consumes the indexed per-pixel and per-sample stream, carries the four-lane spectral
  packet through every stage, and resolves each FrameScene Lambertian material into an inline
  single-component `ClosureSet` and explicit `ClosureMixture`. It samples point, directional, spot,
  and derived emissive-mesh lights and applies explicitly selected balance or power MIS to NEE and
  BSDF-hit contributions. Invalid path slots, queue overflow, incomplete sampling state,
  unsupported media or heuristics, worker failures, and non-Embree acceleration are explicit batch
  errors; the implementation never retries through another backend or transport loop. The scalar
  scene path remains the independent readable oracle. Its versioned CPU report records, for each of
  the seven queues, fixed capacity, peak size, dispatch and lane counts, derived peak and mean
  occupancy, rejected overflow attempts, and accumulated wall-clock nanoseconds around scheduler
  validation, worker execution, and the completion barrier. Timings are observational and excluded
  from deterministic comparisons. Other implemented closure models that are not yet representable
  by the FrameScene material schema remain standalone and are not substituted for Lambertian scene
  materials.
- **Shading normals:** scalar surface transport builds closure frames and evaluates their cosine and
  directional PDF from `Ns`, while `Ng` remains authoritative for sidedness, visibility, ray
  offsets, emission orientation, and area/solid-angle Jacobians. Directions on which the two normals
  disagree have zero support instead of being face-forwarded. The documented Veach adjoint factor
  is exactly one for the current radiance paths and the explicit `|wo.Ns| |wi.Ng| / (|wo.Ng|
  |wi.Ns|)` ratio in importance mode; invalid modes and unrepresentable ratios fail explicitly.
- **BSDF conventions:** a versioned machine-readable contract fixes the numeric PDF measures and
  lobe bits shared by every backend. Concrete surface events contain exactly one
  diffuse/glossy/specular family and one reflection/transmission direction; specular is the delta
  family and volume remains a separate event. Continuous BSDF PDFs use solid angle, delta samples
  use discrete probability, and internal component selection is included exactly once. `wo` and
  `wi` are unit local-closure directions pointing away from the surface toward the previous and
  next path vertices. BSDF values exclude cosine and PDF factors. Radiance and importance modes are
  distinct fixed-width values; only non-symmetric transmission receives the documented radiance
  adjoint factor. Unsupported manifest versions fail instead of returning another schema.
- **Fresnel interfaces:** exact unpolarized dielectric reflectance accepts an explicit
  incident-cosine magnitude and explicitly ordered incident/transmitted refractive indices. The
  conductor evaluator consumes four-lane relative complex indices `eta + i*k` and evaluates the
  complete complex equations independently at every transported wavelength. Both paths support
  transport and reference precision; invalid or unrepresentable values fail instead of being
  clamped, face-forwarded, swapped, or replaced by a Schlick approximation.
- **GGX microfacets:** the isotropic Trowbridge--Reitz distribution exposes projected-normal `D`,
  stable Smith `Lambda` and `G1`, height-correlated `G2`, and exact Heitz visible-normal sampling.
  `alpha` is the finite positive slope width with no perceptual remap; zero is rejected because it
  changes the continuous distribution into a delta. VNDF probabilities are explicitly densities
  over microfacet normals in solid angle. Projected and visible normalization plus deterministic
  normal and oblique chi-square tests cover both transport and reference precision.
- **Rough conductor reflection:** the four-lane isotropic Cook--Torrance lobe combines exact
  conductor Fresnel with the GGX distribution and height-correlated Smith masking-shadowing.
  Reflected directions use Heitz visible-normal sampling and the explicit half-vector Jacobian;
  samples outside the one-sided support remain absent instead of selecting another distribution.
  Float and double quadrature verify reciprocity and white-furnace energy conservation across
  spectral indices, roughness, and view angle. This standalone lobe does not change the current
  Lambertian scene loops.
- **Rough diffuse reflection:** the four-lane energy-preserving Oren--Nayar model combines the
  reciprocal Fujii single-scattering lobe with exact analytical multiple-scattering compensation.
  Its normalized roughness is explicit in `[0, 1]`, zero roughness reduces exactly to Lambert,
  and cosine-weighted samples retain a complete solid-angle PDF. Float and double white-furnace
  quadrature verify unit reflected energy for white surfaces across roughness and view angle.
- **Specular delta lobes:** ideal two-sided mirror reflection and achromatic specular transmission
  expose zero ordinary directional values and densities while sampled atoms use the discrete BSDF
  measure. Transmission follows Snell's law, reports total internal reflection as absent support,
  and returns an explicit mode-aware `etaScale` multiplier reciprocal to the radiance adjoint.
  Unit-vector roundoff introduced by a local frame is reconstructed deterministically; invalid or
  unrepresentable directions, coefficients, indices, and scaling factors fail without an arbitrary
  direction substitute. These standalone lobes do not change the current Lambertian scene loops.
- **Closure storage:** `ClosureSet` owns at most eight insertion-ordered closure records inline
  without dynamic allocation. Transport and reference records retain float and double
  spectral coefficients respectively; their records are fixed at 64 and 120 bytes, and the
  corresponding sets at 520 and 968 bytes, all with eight-byte alignment. Public append operations
  accept validated Lambertian, rough-diffuse, or rough-conductor reflection. Rough diffuse uses the
  first reserved scalar for normalized roughness; rough conductor stores its four-lane relative
  `eta`, four-lane relative `k`, and isotropic `alpha` in nine reserved scalars. Neither model
  changes the record layout. Invalid payloads and a ninth closure return distinct statuses without
  clamping, merging, eviction, or implicit replacement.
  `ClosureMixture` keeps caller-supplied component probabilities in a fixed-capacity dyadic CDF; it
  never derives a scalar probability from spectral weights. Near-unit inputs are projected
  deterministically onto the scalar sampler grid, while invalid or unrepresentable distributions
  fail explicitly. Physical closure values add directly, and the solid-angle directional PDF is
  the explicit sum `p_mix = sum(q_i * p_i)`. Sampling reports the selected slot and its effective
  discrete probability, and returns the full mixture value and PDF without applying selection
  twice.
- **Sampling:** indexed `SampleStream` values with a versioned dimension map, independent hashing,
  local PCG32, stratification, Latin hypercube, high-dimensional Sobol, reproducible Owen
  scrambling, common disk/sphere/hemisphere mappings, and immutable uniform or spectral-power
  light-selection distributions plus context-dependent Light Tree selection. The CUDA C++20 path
  reproduces the complete version-one camera and bounce dimension map, rejects unsupported schemas
  or unrepresentable bounce indices, and emits a canonical device sample dump compared bit-for-bit
  with the independent CPU implementation. No global mutable RNG is used.
- **Spectral and color foundations:** four-lane 360-830 nm wavelength packets with marginal PDFs,
  black, constant, and tabulated spectra, `SampledSpectrum<4>`, energy-conserving spectral
  Lambertian and energy-preserving rough-diffuse reflection, one-sided spectral surface emission,
  a constant spectral environment, the CIE 1931 2-degree observer to relative XYZ, and signed
  scene-linear RGB.
- **Light contract:** statically dispatched float and double light models expose incident-radiance
  sampling, directional PDFs, escaped-ray radiance, spectral power, and conservative world bounds.
  Continuous and delta probabilities retain distinct solid-angle and discrete measures. Explicit
  area-to-solid-angle and solid-angle-to-area PDF conversions use the geometric-normal Jacobian and
  reject singular or unrepresentable configurations without clamping or measure substitution.
  Packet-bound ideal point, directional, and spot lights provide strict inverse-square, cone
  falloff, and scene-bounds power conventions. Packet-bound rectangle, disk, sphere, and compact
  indexed-mesh area lights sample surface area, keep sampled PDFs bound to their explicit endpoint,
  query the closest surface along a direction, expose explicit one- and two-sided emission, and
  report conservative finite support bounds. Packet-bound latitude-longitude environment maps
  support explicit quaternion rotation and a normalized two-dimensional importance distribution
  built from spectral packet radiance and exact texel solid angles. The light sampler selects stable
  registry slots uniformly, by normalized four-lane spectral power, or through an immutable
  spatial Light Tree. The tree builds a deterministic median BVH over finite emitters, isolates
  canonically unbounded lights, and normalizes a context-dependent power-and-distance heuristic
  without changing registry slots. Its discrete probability remains separate from each selected
  light's conditional PDF. Scalar scene paths connect this selection contract to both the
  frame-scene punctual registry and its derived emissive-mesh registry through robust
  closest-hit/occlusion backends. Environment models remain hit-only in the current transport path.
- **Film and output:** weighted float accumulation, compensated double reference accumulation,
  crops, deterministic tile fusion, a tested scene-linear 32-bit RGB OpenEXR writer, and an
  optional stb-backed PNG preview writer with a fixed display transform.
- **Validation:** linear and HDR error metrics, display-referred PSNR and heatmaps, plus debug
  encodings for normals, depth, UVs, barycentrics, and identifiers covered by five 64x64 goldens.
  A deterministic tilted-normal Lambertian furnace validates both precisions against its analytic
  common-hemisphere energy and verifies that the correction does not create energy.
  A public checkpoint contract reports the first observed inclusive time-to-MSE and time-to-PSNR
  crossings without interpolation or a last-checkpoint substitute. A Release benchmark renders one
  canonical 128x128 Cornell box through FrameScene, Embree, spectral Lambertian transport,
  next-event estimation, and power-heuristic MIS. The scalar scene path creates its independent
  512-spp oracle image; the four-worker CPU wavefront path creates eight seeded progressive
  evaluations and the canonical 256-spp PNG: red and green walls, neutral enclosure and ceiling
  emitter, and exactly two white Lambertian triangle-mesh spheres. Its machine-readable JSON report
  records the wavefront worker count, median and median absolute deviation for time-to-quality and
  final error, checks the worst seed against fixed MSE/PSNR targets, and validates coarse
  linear-image semantics; a missed target is an explicit benchmark failure.
  A separate one-sample Cornell benchmark emits a machine-readable queue report for 16,384 primary
  paths and validates all seven capacities, peaks, occupations, overflow counters, flow identities,
  and per-stage wall times without imposing a fragile performance threshold.
  A versioned same-sample parity matrix compares the scalar scene loop with analytic traversal and
  compensated double film accumulation against the four-worker CPU wavefront loop with Embree and
  float film accumulation. Its nine-entry inventory covers balance and power area-light transport,
  environment misses, Russian roulette, subnormal throughput, occluded extreme radiometry,
  point-light NEE, Veach MIS, and the canonical two-sphere Cornell scene. Every entry uses float
  transport and sends identical indexed rays, wavelengths, samples, and seed to both backends. The
  consolidated JSON report requires linear MSE at most `1e-10`, RMSE at most `1e-5`, image and
  per-path absolute error at most `1e-4`, and display PSNR of at least 80 dB (or positive infinity).
  It also verifies the explicit analytic/Embree backend kinds and rejects queue overflow or rejected
  lanes; neither side can be substituted by a fallback.
  Two closed CornellDiffuse validation fixtures provide tracked 64x64 and 256x256 scene-linear EXR
  references rendered by `scalar_ref` at 4096 and 1024 spp. Their scenes, generator source snapshot,
  and image hashes are verified before deterministic 1-versus-4-spp MSE, RMSE, and display-PSNR
  convergence checks. A separate 64x64 regression rebuilds the Cornell fixture as five immutable
  frame-scene meshes and materials, traverses all path rays through the explicit Embree backend,
  and compares MSE, RMSE, and display PSNR against the independent double scalar oracle. A distinct
  point-light scene exercises punctual NEE over real Embree closest-hit and shadow traversal,
  verifies deterministic replay, and checks median MSE/PSNR convergence over eight seeds at 1, 4,
  and 16 spp. Its 64-spp PNG preview is checksum-locked and checked against an enumerated double
  direct-lighting result. A distinct Veach-style scene uses a Lambertian receiver and a mesh emitter
  through Embree to exercise both direct-light and BSDF-hit estimators. Balance and power estimates
  agree with independent double quadrature within their fixed tolerance and remain close to one
  reference contribution rather than two; replay and dispatch failures are also covered. Its
  checksum-locked neutral 64x64 preview shows four white mesh emitters of different sizes at 128 spp.
- **Host control:** bounded versioned local IPC, a C extension ABI, explicit absolute-path loading,
  XPU device discovery, a reference discovery plugin, and a headless `render` control executable.
- **Backend integration:** explicit capability reporting and pre-dispatch checks, pinned Embree 4
  closest-hit/occlusion queries, and CUDA C++20 closest-hit and opaque any-hit kernels. The
  host/device transport boundary is isolated in a versioned C++20-compatible header containing
  fixed-width, explicitly
  aligned ray, hit, queue, sampler, spectrum, and path-state records. The host compiler and a CUDA
  kernel compare every frozen size, alignment, and member offset at test time. A separate host-only
  C++26 layer owns typed device buffers through move-only RAII, provides explicitly bounded and
  aligned scratch suballocation, preserves live storage when growth is refused, and reports CUDA
  allocation exhaustion without selecting host memory. A committed `FrameScene` serializes into a
  deterministic pointer-free device SoA, and a separate versioned CUDA blob builds one binary BLAS
  per geometry plus a TLAS over resolved instances with conservative finite bounds. CUDA scene
  queries traverse that TLAS and its BLAS, apply resolved instance transforms and intersect
  two-sided watertight triangles with explicit per-ray miss and error states. Opaque shadow queries
  apply visibility masks and exit at the first eligible crossing without reconstructing surface
  data. Seven fixed-capacity CUDA queue columns now store path-slot indices for camera, ray, hit,
  miss, shade, shadow, and continuation work. Device-side CAS reservation keeps each published size
  within capacity, while saturated overflow and rejection counters make exhaustion explicit without
  overwriting adjacent storage; reset requires an explicit overflow policy. Six CUDA C++20 stage
  kernels now consume those queues for camera initialization, hit resolution, miss handling,
  Lambertian shading, shadow visibility, and continuation. The host orchestrator dispatches the
  existing CUDA closest-hit and any-hit kernels between stages, validates every per-lane outcome and
  queue boundary, and reconstructs results in input path-slot order. The device shading path reads
  four-lane reflectance, one-sided emission, constant environment radiance, and mesh-area lights
  directly from the serialized scene. Its light and BSDF samples use named dimensions from the
  complete CUDA `SampleStream` contract rather than private numeric offsets. Embree stays an
  independent CPU oracle and is never substituted for CUDA execution.

## Current boundary

Blackframe does not yet provide a general production path-tracing integrator, a scene loader, a
general material or lighting system, deformation updates, or transform motion blur. The CPU
wavefront transport is currently a narrow vacuum-only spectral Lambertian path over FrameScene and
Embree, with punctual and emissive triangle-mesh lights, NEE, and balance or power MIS. The CUDA
wavefront transport currently covers vacuum Lambertian paths, one-sided emissive triangle meshes,
uniform mesh-area NEE, a constant environment, and bounded diffuse depth. It rejects punctual
lights, participating media, unsupported scene spectra, and malformed device state explicitly;
its current direct-light estimator avoids duplicate emissive paths without claiming MIS or Russian
roulette support. `scalar_ref` remains a separate oracle rather than an execution fallback.
The rough diffuse, rough conductor, specular delta, rough dielectric, and anisotropic closure
implementations remain standalone until the FrameScene material schema can describe them.
Environment maps are not yet sampled by NEE/MIS, and the public MIS entry points start complete
primary paths because `PathState` does not carry a prior vertex's directional PDFs. Acceleration
updates currently cover explicit full rebuilds and frame-to-frame transform refits between
immutable snapshots only. The headless executable currently supports local engine control and
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
  architectures must be explicit, and the supplied presets target architecture 86. CUDA test
  configurations also require the matching `compute-sanitizer` shipped by that toolkit.

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

The canonical Cornell convergence report and PNG are generated and validated together in a Release
build:

```powershell
ctest --test-dir build/windows-cpu-release `
  -R '^Blackframe\.Benchmarks\.CornellPowerMisConvergence\.Json$' `
  --output-on-failure
```

The complete scalar-to-CPU-wavefront parity inventory and its machine-readable report are validated
with:

```powershell
ctest --test-dir build/windows-cpu-release `
  -R '^Blackframe\.EmbreeScalarWavefrontParity\.Json$' `
  --output-on-failure
```

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
