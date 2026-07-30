#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
using SurfaceEmissionFor =
    std::conditional_t<std::is_same_v<Scalar, TransportScalar>, OneSidedSurfaceEmission,
                       ReferenceOneSidedSurfaceEmission>;

template <SpectrumScalar Scalar>
using EnvironmentFor = std::conditional_t<std::is_same_v<Scalar, TransportScalar>,
                                          ConstantEnvironment, ReferenceConstantEnvironment>;

inline constexpr std::string_view SurfaceRadianceError =
    "One-sided surface emission requires every spectral lane to be finite and non-negative.";
inline constexpr std::string_view EnvironmentRadianceError =
    "Constant environment radiance requires every spectral lane to be finite and non-negative.";
inline constexpr std::string_view SurfaceDirectionError =
    "Surface emission evaluation requires a finite unit geometric normal and a finite non-zero "
    "outgoing direction.";
inline constexpr std::string_view SurfaceOrientationError =
    "Surface emission orientation is not representable.";
inline constexpr std::string_view EnvironmentDirectionError =
    "Constant environment evaluation requires a finite non-zero direction.";

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> surface_radiance() {
    return {
        .values =
            {
                Scalar{0.25},
                Scalar{1},
                Scalar{2},
                Scalar{8},
            },
    };
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> sky_radiance() {
    return {
        .values =
            {
                Scalar{0.125},
                Scalar{0.5},
                Scalar{1.5},
                Scalar{4},
            },
    };
}

template <typename Result>
void expect_invalid(const Result& result, const std::string_view expected_message) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, expected_message);
}

template <SpectrumScalar Scalar> void expect_emission_creation_contract() {
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto hdr = SpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{0.5}, Scalar{2}, maximum},
    };
    const auto surface = SurfaceEmissionFor<Scalar>::create(hdr);
    const auto environment = EnvironmentFor<Scalar>::create(hdr);
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());
    EXPECT_EQ(surface->radiance(), hdr);
    EXPECT_EQ(environment->radiance(), hdr);

    const auto signed_zero = SpectrumFor<Scalar>{
        .values = {-Scalar{0}, Scalar{0}, Scalar{1}, Scalar{2}},
    };
    EXPECT_TRUE(SurfaceEmissionFor<Scalar>::create(signed_zero).has_value());
    EXPECT_TRUE(EnvironmentFor<Scalar>::create(signed_zero).has_value());

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto below_zero = std::nextafter(Scalar{0}, -infinity);
    for (const auto invalid :
         std::array{below_zero, std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto malformed = hdr;
            malformed[lane] = invalid;
            expect_invalid(SurfaceEmissionFor<Scalar>::create(malformed), SurfaceRadianceError);
            expect_invalid(EnvironmentFor<Scalar>::create(malformed), EnvironmentRadianceError);
        }
    }
}

TEST(EmissionTest, CreatesFiniteNonNegativeHdrRadianceWithoutClamping) {
    static_assert(std::same_as<decltype(OneSidedSurfaceEmission::create(TransportSpectrum{})),
                               core::Result<OneSidedSurfaceEmission>>);
    static_assert(
        std::same_as<decltype(ReferenceOneSidedSurfaceEmission::create(ReferenceSpectrum{})),
                     core::Result<ReferenceOneSidedSurfaceEmission>>);
    static_assert(std::same_as<decltype(ConstantEnvironment::create(TransportSpectrum{})),
                               core::Result<ConstantEnvironment>>);
    static_assert(std::same_as<decltype(ReferenceConstantEnvironment::create(ReferenceSpectrum{})),
                               core::Result<ReferenceConstantEnvironment>>);
    static_assert(std::same_as<decltype(std::declval<const OneSidedSurfaceEmission&>().eval(
                                   std::declval<Normal3>(), std::declval<Vector3>())),
                               core::Result<TransportSpectrum>>);
    static_assert(std::same_as<decltype(std::declval<const ReferenceConstantEnvironment&>().eval(
                                   std::declval<ReferenceVector3>())),
                               core::Result<ReferenceSpectrum>>);
    static_assert(!std::same_as<OneSidedSurfaceEmission, ReferenceOneSidedSurfaceEmission>);
    static_assert(!std::same_as<ConstantEnvironment, ReferenceConstantEnvironment>);

    expect_emission_creation_contract<TransportScalar>();
    expect_emission_creation_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_surface_and_environment_evaluation() {
    const auto emitted = surface_radiance<Scalar>();
    const auto sky = sky_radiance<Scalar>();
    const auto surface = SurfaceEmissionFor<Scalar>::create(emitted);
    const auto environment = EnvironmentFor<Scalar>::create(sky);
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());

    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto rounded_normal = Normal3T<Scalar>{
        .z = std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
    };
    const auto maximum = std::numeric_limits<Scalar>::max();
    for (const auto outgoing : std::array{
             Vector3T<Scalar>{.z = Scalar{1}},
             Vector3T<Scalar>{.x = Scalar{3}, .z = Scalar{4}},
             Vector3T<Scalar>{.x = maximum, .z = maximum},
             Vector3T<Scalar>{.z = std::numeric_limits<Scalar>::denorm_min()},
         }) {
        const auto evaluated = surface->eval(normal, outgoing);
        ASSERT_TRUE(evaluated.has_value());
        EXPECT_EQ(*evaluated, emitted);
    }
    const auto rounded = surface->eval(rounded_normal, Vector3T<Scalar>{.z = Scalar{1}});
    ASSERT_TRUE(rounded.has_value());
    EXPECT_EQ(*rounded, emitted);

    const auto minimum = std::numeric_limits<Scalar>::min();
    expect_invalid(surface->eval(Normal3T<Scalar>{.x = Scalar{1}, .y = minimum},
                                 Vector3T<Scalar>{.y = minimum, .z = Scalar{1}}),
                   SurfaceOrientationError);
    expect_invalid(surface->eval(normal,
                                 Vector3T<Scalar>{
                                     .x = maximum,
                                     .z = std::numeric_limits<Scalar>::denorm_min(),
                                 }),
                   SurfaceOrientationError);

    for (const auto outgoing : std::array{
             Vector3T<Scalar>{.x = Scalar{1}},
             Vector3T<Scalar>{.x = Scalar{1}, .z = -Scalar{0}},
             Vector3T<Scalar>{.z = Scalar{-1}},
             Vector3T<Scalar>{.x = Scalar{3}, .z = Scalar{-4}},
             Vector3T<Scalar>{.x = maximum, .z = -maximum},
             Vector3T<Scalar>{.z = -std::numeric_limits<Scalar>::denorm_min()},
         }) {
        const auto evaluated = surface->eval(normal, outgoing);
        ASSERT_TRUE(evaluated.has_value());
        EXPECT_EQ(*evaluated, SpectrumFor<Scalar>{});
    }

    const auto inverted =
        surface->eval(Normal3T<Scalar>{.z = Scalar{-1}}, Vector3T<Scalar>{.z = Scalar{-2}});
    ASSERT_TRUE(inverted.has_value());
    EXPECT_EQ(*inverted, emitted);

    for (const auto direction : std::array{
             Vector3T<Scalar>{.z = Scalar{1}},
             Vector3T<Scalar>{.z = Scalar{-2}},
             Vector3T<Scalar>{.x = maximum, .y = -maximum, .z = maximum},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::denorm_min()},
         }) {
        const auto evaluated = environment->eval(direction);
        ASSERT_TRUE(evaluated.has_value());
        EXPECT_EQ(*evaluated, sky);
    }
}

TEST(EmissionTest, EvaluatesOneSidedSurfaceAndDirectionIndependentEnvironment) {
    expect_surface_and_environment_evaluation<TransportScalar>();
    expect_surface_and_environment_evaluation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_evaluation_inputs_rejected() {
    const auto surface = SurfaceEmissionFor<Scalar>::create(surface_radiance<Scalar>());
    const auto environment = EnvironmentFor<Scalar>::create(sky_radiance<Scalar>());
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());

    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto direction = Vector3T<Scalar>{.z = Scalar{1}};
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    for (const auto invalid_normal : std::array{
             Normal3T<Scalar>{},
             Normal3T<Scalar>{.z = Scalar{2}},
             Normal3T<Scalar>{
                 .z = Scalar{1} + Scalar{256} * std::numeric_limits<Scalar>::epsilon(),
             },
             Normal3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
             Normal3T<Scalar>{.y = infinity, .z = Scalar{1}},
             Normal3T<Scalar>{.x = -infinity, .z = Scalar{1}},
         }) {
        expect_invalid(surface->eval(invalid_normal, direction), SurfaceDirectionError);
    }

    for (const auto invalid_direction : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{-1}},
             Vector3T<Scalar>{.y = infinity, .z = Scalar{-1}},
             Vector3T<Scalar>{.x = -infinity, .z = Scalar{-1}},
         }) {
        expect_invalid(surface->eval(normal, invalid_direction), SurfaceDirectionError);
        expect_invalid(environment->eval(invalid_direction), EnvironmentDirectionError);
    }
}

TEST(EmissionTest, RejectsMalformedEvaluationInputsWithoutSubstitution) {
    expect_invalid_evaluation_inputs_rejected<TransportScalar>();
    expect_invalid_evaluation_inputs_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>> make_visibility_ray(const Point3T<Scalar> origin,
                                                             const Vector3T<Scalar> direction) {
    return RayT<Scalar>::create(origin, direction, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), Scalar{0},
                                AllRayVisibility, VacuumMedium);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::array<TriangleT<Scalar>, 2>>
make_emissive_quad(const bool reverse_winding) {
    const auto lower_left = Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}};
    const auto lower_right = Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-1}};
    const auto upper_right = Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}};
    const auto upper_left = Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{1}};

    const auto first = reverse_winding
                           ? TriangleT<Scalar>::create(lower_left, upper_right, lower_right)
                           : TriangleT<Scalar>::create(lower_left, lower_right, upper_right);
    const auto second = reverse_winding
                            ? TriangleT<Scalar>::create(lower_left, upper_left, upper_right)
                            : TriangleT<Scalar>::create(lower_left, upper_right, upper_left);
    if (!first.has_value()) {
        return std::unexpected(first.error());
    }
    if (!second.has_value()) {
        return std::unexpected(second.error());
    }
    return std::array{*first, *second};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SpectrumFor<Scalar>>
trace_visible_emission(const RayT<Scalar>& ray, const std::array<TriangleT<Scalar>, 2>& quad,
                       const SurfaceEmissionFor<Scalar>& surface,
                       const EnvironmentFor<Scalar>& environment) {
    auto nearest = std::optional<TriangleHitT<Scalar>>{};
    for (const auto& triangle : quad) {
        const auto intersection = triangle.intersect(ray);
        if (!intersection.has_value()) {
            return std::unexpected(intersection.error());
        }
        if (intersection->has_value() &&
            (!nearest.has_value() || (**intersection).parameter < nearest->parameter)) {
            nearest = **intersection;
        }
    }

    if (nearest.has_value()) {
        return surface.eval(nearest->geometric_normal, -ray.direction());
    }
    return environment.eval(ray.direction());
}

template <SpectrumScalar Scalar> void expect_quad_and_sky_visibility() {
    const auto quad = make_emissive_quad<Scalar>(false);
    const auto reversed_quad = make_emissive_quad<Scalar>(true);
    const auto surface = SurfaceEmissionFor<Scalar>::create(surface_radiance<Scalar>());
    const auto black_surface = SurfaceEmissionFor<Scalar>::create(SpectrumFor<Scalar>{});
    const auto environment = EnvironmentFor<Scalar>::create(sky_radiance<Scalar>());
    ASSERT_TRUE(quad.has_value());
    ASSERT_TRUE(reversed_quad.has_value());
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(black_surface.has_value());
    ASSERT_TRUE(environment.has_value());

    for (const auto target : std::array{
             Point3T<Scalar>{.x = Scalar{0.5}, .y = Scalar{-0.5}, .z = Scalar{1}},
             Point3T<Scalar>{.x = Scalar{-0.5}, .y = Scalar{0.5}, .z = Scalar{1}},
             Point3T<Scalar>{.z = Scalar{1}},
         }) {
        const auto ray = make_visibility_ray(target, Vector3T<Scalar>{.z = Scalar{-1}});
        ASSERT_TRUE(ray.has_value());
        const auto visible = trace_visible_emission(*ray, *quad, *surface, *environment);
        ASSERT_TRUE(visible.has_value());
        EXPECT_EQ(*visible, surface_radiance<Scalar>());
    }

    const auto back_ray =
        make_visibility_ray(Point3T<Scalar>{.z = Scalar{-1}}, Vector3T<Scalar>{.z = Scalar{1}});
    ASSERT_TRUE(back_ray.has_value());
    const auto back = trace_visible_emission(*back_ray, *quad, *surface, *environment);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, SpectrumFor<Scalar>{});
    EXPECT_NE(*back, sky_radiance<Scalar>());

    const auto front_ray =
        make_visibility_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}});
    ASSERT_TRUE(front_ray.has_value());
    const auto reversed =
        trace_visible_emission(*front_ray, *reversed_quad, *surface, *environment);
    const auto black_hit = trace_visible_emission(*front_ray, *quad, *black_surface, *environment);
    ASSERT_TRUE(reversed.has_value());
    ASSERT_TRUE(black_hit.has_value());
    EXPECT_EQ(*reversed, SpectrumFor<Scalar>{});
    EXPECT_EQ(*black_hit, SpectrumFor<Scalar>{});

    for (const auto ray : std::array{
             make_visibility_ray(Point3T<Scalar>{.x = Scalar{2}, .z = Scalar{1}},
                                 Vector3T<Scalar>{.z = Scalar{-1}}),
             make_visibility_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{1}}),
         }) {
        ASSERT_TRUE(ray.has_value());
        const auto visible = trace_visible_emission(*ray, *quad, *surface, *environment);
        ASSERT_TRUE(visible.has_value());
        EXPECT_EQ(*visible, sky_radiance<Scalar>());
    }

    const auto coplanar = make_visibility_ray(Point3T<Scalar>{}, Vector3T<Scalar>{.x = Scalar{1}});
    ASSERT_TRUE(coplanar.has_value());
    const auto invalid = trace_visible_emission(*coplanar, *quad, *surface, *environment);
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(invalid.error().message, std::string_view{});
}

TEST(EmissionTest, MakesQuadHitsOccludeTheConstantSky) {
    expect_quad_and_sky_visibility<TransportScalar>();
    expect_quad_and_sky_visibility<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
