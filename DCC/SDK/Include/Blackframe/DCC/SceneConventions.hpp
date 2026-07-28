#pragma once

#include <cstdint>

namespace blackframe::dcc {

enum class Handedness : std::uint8_t {
    right_handed,
    left_handed,
};

enum class UpAxis : std::uint8_t {
    positive_x,
    positive_y,
    positive_z,
};

struct SceneConventions {
    double meters_per_distance_unit{1.0};
    double seconds_per_time_unit{1.0};
    Handedness handedness{Handedness::right_handed};
    UpAxis up_axis{UpAxis::positive_y};
};

} // namespace blackframe::dcc
