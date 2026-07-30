#include <Blackframe/Engine/AccelBackend.hpp>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error accel_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

class AnalyticAccelBackend final : public AccelBackend {
  public:
    [[nodiscard]] static core::Result<std::unique_ptr<AccelBackend>>
    create(const std::span<const AccelGeometry> geometries) {
        const auto validation = validate_geometry_input(geometries);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        try {
            auto retained_geometries =
                std::vector<AccelGeometry>{geometries.begin(), geometries.end()};
            return std::unique_ptr<AccelBackend>{
                new AnalyticAccelBackend{std::move(retained_geometries)}};
        } catch (const std::bad_alloc&) {
            return std::unexpected(
                accel_error(core::StatusCode::resource_exhausted,
                            "Analytic acceleration construction exhausted host memory."));
        } catch (const std::length_error&) {
            return std::unexpected(
                accel_error(core::StatusCode::resource_exhausted,
                            "Analytic acceleration exceeded host container limits."));
        }
    }

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return AccelBackendKind::analytic_reference;
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        auto closest = std::optional<AccelHit>{};
        for (const auto& geometry : geometries_) {
            if ((ray.mask() & geometry.visibility_mask) == 0U) {
                continue;
            }

            for (auto triangle_index = std::size_t{};
                 triangle_index < geometry.mesh->triangles().size(); ++triangle_index) {
                auto triangle = geometry.mesh->geometric_triangle(triangle_index);
                if (!triangle) {
                    return std::unexpected(triangle.error());
                }
                auto candidate = triangle->intersect_classified(ray);
                if (!candidate) {
                    if (candidate.error().kind ==
                        renderer::TriangleIntersectionErrorKind::coplanar_ambiguity) {
                        continue;
                    }
                    return std::unexpected(std::move(candidate.error().diagnostic));
                }
                if (!candidate->has_value()) {
                    continue;
                }
                if (closest.has_value() &&
                    closest->triangle.parameter <= candidate->value().parameter) {
                    continue;
                }

                closest = AccelHit{
                    .triangle = candidate->value(),
                    .identifiers =
                        {
                            .instance = geometry.instance,
                            .geometry = geometry.geometry,
                            .primitive =
                                renderer::PrimitiveId{
                                    .value = static_cast<std::uint32_t>(triangle_index),
                                },
                            .material = geometry.material,
                        },
                };
            }
        }
        return closest;
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        for (const auto& geometry : geometries_) {
            if ((ray.mask() & geometry.visibility_mask) == 0U) {
                continue;
            }

            for (auto triangle_index = std::size_t{};
                 triangle_index < geometry.mesh->triangles().size(); ++triangle_index) {
                auto triangle = geometry.mesh->geometric_triangle(triangle_index);
                if (!triangle) {
                    return std::unexpected(triangle.error());
                }
                auto candidate = triangle->intersect_classified(ray);
                if (!candidate) {
                    if (candidate.error().kind ==
                        renderer::TriangleIntersectionErrorKind::coplanar_ambiguity) {
                        continue;
                    }
                    return std::unexpected(std::move(candidate.error().diagnostic));
                }
                if (candidate->has_value()) {
                    return true;
                }
            }
        }
        return false;
    }

  private:
    explicit AnalyticAccelBackend(std::vector<AccelGeometry>&& geometries) noexcept
        : geometries_{std::move(geometries)} {}

    std::vector<AccelGeometry> geometries_;
};

} // namespace

core::Status
AccelBackend::validate_geometry_input(const std::span<const AccelGeometry> geometries) {
    for (auto geometry_index = std::size_t{}; geometry_index < geometries.size();
         ++geometry_index) {
        const auto& geometry = geometries[geometry_index];
        if (!geometry.mesh) {
            return std::unexpected(
                accel_error(core::StatusCode::invalid_argument,
                            "Acceleration geometry requires an immutable triangle mesh."));
        }

        for (auto previous_index = std::size_t{}; previous_index < geometry_index;
             ++previous_index) {
            const auto& previous = geometries[previous_index];
            if (geometry.instance == previous.instance && geometry.geometry == previous.geometry) {
                return std::unexpected(accel_error(
                    core::StatusCode::invalid_argument,
                    "Acceleration geometry identities must be unique within a backend."));
            }
        }
    }
    return {};
}

core::Status AccelBackend::validate_ray(const renderer::Ray& ray) {
    if (ray.time() < renderer::TransportScalar{0} || ray.time() > renderer::TransportScalar{1}) {
        return std::unexpected(
            accel_error(core::StatusCode::invalid_argument,
                        "Acceleration queries require ray time in the normalized [0, 1] range."));
    }
    return {};
}

core::Result<std::unique_ptr<AccelBackend>>
create_analytic_accel_backend(const std::span<const AccelGeometry> geometries) {
    return AnalyticAccelBackend::create(geometries);
}

} // namespace blackframe::engine
