#include "RenderConfigurationLimits.hpp"

#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace blackframe::renderer {
namespace {

constexpr auto configuration_fields = std::array{
    RenderConfigurationFieldSchema{
        .key = "schema_version",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = true,
        .minimum_unsigned_value = CurrentRenderConfigurationSchemaVersion,
        .maximum_unsigned_value = CurrentRenderConfigurationSchemaVersion,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "width",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 1,
        .maximum_unsigned_value = limits::maximum_dimension,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "height",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 1,
        .maximum_unsigned_value = limits::maximum_dimension,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "samples_per_pixel",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 1,
        .maximum_unsigned_value = limits::maximum_samples_per_pixel,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "maximum_path_depth",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 1,
        .maximum_unsigned_value = limits::maximum_path_depth,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "tile_edge_length",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 1,
        .maximum_unsigned_value = limits::maximum_tile_edge_length,
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "seed",
        .value_kind = RenderConfigurationValueKind::unsigned_integer,
        .required = false,
        .minimum_unsigned_value = 0,
        .maximum_unsigned_value = std::numeric_limits<std::uint64_t>::max(),
        .maximum_string_size = 0,
    },
    RenderConfigurationFieldSchema{
        .key = "xpu_device_id",
        .value_kind = RenderConfigurationValueKind::string,
        .required = false,
        .minimum_unsigned_value = 0,
        .maximum_unsigned_value = 0,
        .maximum_string_size = limits::maximum_xpu_device_id_size,
    },
};

[[nodiscard]] core::Error configuration_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

class JsonCursor final {
  public:
    explicit JsonCursor(const std::string_view input) noexcept : input_{input} {}

    [[nodiscard]] bool consume(const char expected) noexcept {
        skip_whitespace();
        if (offset_ >= input_.size() || input_[offset_] != expected) {
            return false;
        }
        ++offset_;
        return true;
    }

    [[nodiscard]] bool finished() noexcept {
        skip_whitespace();
        return offset_ == input_.size();
    }

    [[nodiscard]] core::Result<std::string> parse_string(const std::size_t maximum_size,
                                                         const std::string_view description) {
        skip_whitespace();
        if (offset_ >= input_.size() || input_[offset_] != '"') {
            return std::unexpected(syntax_error("Expected " + std::string{description} + "."));
        }
        ++offset_;

        std::string output;
        while (offset_ < input_.size()) {
            const auto character = input_[offset_++];
            if (character == '"') {
                return output;
            }
            if (static_cast<unsigned char>(character) < 0x20U) {
                return std::unexpected(syntax_error("JSON strings cannot contain control bytes."));
            }
            if (character != '\\') {
                if (output.size() == maximum_size) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                output.push_back(character);
                continue;
            }

            if (offset_ >= input_.size()) {
                return std::unexpected(syntax_error("Unterminated JSON escape sequence."));
            }
            const auto escape = input_[offset_++];
            switch (escape) {
            case '"':
            case '\\':
            case '/':
                if (output.size() == maximum_size) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                output.push_back(escape);
                break;
            case 'b':
                if (!append_byte(output, '\b', maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            case 'f':
                if (!append_byte(output, '\f', maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            case 'n':
                if (!append_byte(output, '\n', maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            case 'r':
                if (!append_byte(output, '\r', maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            case 't':
                if (!append_byte(output, '\t', maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            case 'u': {
                auto code_point = parse_unicode_escape();
                if (!code_point) {
                    return std::unexpected(std::move(code_point.error()));
                }
                if (!append_utf8(output, *code_point, maximum_size)) {
                    return std::unexpected(size_error(description, maximum_size));
                }
                break;
            }
            default:
                return std::unexpected(syntax_error("Unknown JSON escape sequence."));
            }
        }

        return std::unexpected(syntax_error("Unterminated JSON string."));
    }

    [[nodiscard]] core::Result<std::uint64_t> parse_unsigned_integer(const std::string_view key) {
        skip_whitespace();
        const auto start = offset_;
        while (offset_ < input_.size() && !is_value_delimiter(input_[offset_])) {
            ++offset_;
        }
        const auto token = input_.substr(start, offset_ - start);

        std::uint64_t value = 0;
        const auto conversion = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || conversion.ec != std::errc{} ||
            conversion.ptr != token.data() + token.size() ||
            (token.size() > 1 && token.front() == '0')) {
            return std::unexpected(configuration_error(core::StatusCode::invalid_argument,
                                                       "Render configuration key '" +
                                                           std::string{key} +
                                                           "' must be an unsigned JSON integer."));
        }
        return value;
    }

    [[nodiscard]] core::Error syntax_error(std::string message) const {
        message += " At byte " + std::to_string(offset_) + ".";
        return configuration_error(core::StatusCode::invalid_argument, std::move(message));
    }

  private:
    static bool append_byte(std::string& output, const char value, const std::size_t maximum_size) {
        if (output.size() == maximum_size) {
            return false;
        }
        output.push_back(value);
        return true;
    }

    static bool append_utf8(std::string& output, const std::uint32_t code_point,
                            const std::size_t maximum_size) {
        auto bytes = std::array<char, 4>{};
        std::size_t byte_count = 0;
        if (code_point <= 0x7FU) {
            bytes[0] = static_cast<char>(code_point);
            byte_count = 1;
        } else if (code_point <= 0x7FFU) {
            bytes[0] = static_cast<char>(0xC0U | (code_point >> 6U));
            bytes[1] = static_cast<char>(0x80U | (code_point & 0x3FU));
            byte_count = 2;
        } else if (code_point <= 0xFFFFU) {
            bytes[0] = static_cast<char>(0xE0U | (code_point >> 12U));
            bytes[1] = static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU));
            bytes[2] = static_cast<char>(0x80U | (code_point & 0x3FU));
            byte_count = 3;
        } else {
            bytes[0] = static_cast<char>(0xF0U | (code_point >> 18U));
            bytes[1] = static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU));
            bytes[2] = static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU));
            bytes[3] = static_cast<char>(0x80U | (code_point & 0x3FU));
            byte_count = 4;
        }

        if (byte_count > maximum_size - output.size()) {
            return false;
        }
        output.append(bytes.data(), byte_count);
        return true;
    }

    [[nodiscard]] core::Result<std::uint32_t> parse_hex_quad() {
        if (input_.size() - offset_ < 4) {
            return std::unexpected(syntax_error("Incomplete JSON Unicode escape."));
        }

        std::uint32_t value = 0;
        for (auto digit_index = 0U; digit_index < 4U; ++digit_index) {
            const auto character = input_[offset_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') {
                value |= static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                value |= static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                value |= static_cast<std::uint32_t>(character - 'A' + 10);
            } else {
                return std::unexpected(syntax_error("Invalid JSON Unicode escape."));
            }
        }
        return value;
    }

    [[nodiscard]] core::Result<std::uint32_t> parse_unicode_escape() {
        auto first = parse_hex_quad();
        if (!first) {
            return first;
        }
        if (*first >= 0xDC00U && *first <= 0xDFFFU) {
            return std::unexpected(syntax_error("Unexpected low Unicode surrogate."));
        }
        if (*first < 0xD800U || *first > 0xDBFFU) {
            return first;
        }

        if (input_.size() - offset_ < 2 || input_[offset_] != '\\' || input_[offset_ + 1] != 'u') {
            return std::unexpected(syntax_error("Missing low Unicode surrogate."));
        }
        offset_ += 2;
        auto second = parse_hex_quad();
        if (!second) {
            return second;
        }
        if (*second < 0xDC00U || *second > 0xDFFFU) {
            return std::unexpected(syntax_error("Invalid low Unicode surrogate."));
        }
        return 0x10000U + ((*first - 0xD800U) << 10U) + (*second - 0xDC00U);
    }

    [[nodiscard]] static core::Error size_error(const std::string_view description,
                                                const std::size_t maximum_size) {
        return configuration_error(core::StatusCode::resource_exhausted,
                                   std::string{description} + " exceeds " +
                                       std::to_string(maximum_size) + " bytes.");
    }

    [[nodiscard]] static constexpr bool is_value_delimiter(const char value) noexcept {
        return value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == ',' ||
               value == '}';
    }

    void skip_whitespace() noexcept {
        while (offset_ < input_.size()) {
            const auto value = input_[offset_];
            if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
                break;
            }
            ++offset_;
        }
    }

    std::string_view input_;
    std::size_t offset_{};
};

[[nodiscard]] std::size_t schema_field_index(const std::string_view key) noexcept {
    for (std::size_t index = 0; index < configuration_fields.size(); ++index) {
        if (configuration_fields[index].key == key) {
            return index;
        }
    }
    return configuration_fields.size();
}

[[nodiscard]] core::Error range_error(const RenderConfigurationFieldSchema& field) {
    return configuration_error(core::StatusCode::invalid_argument,
                               "Render configuration key '" + std::string{field.key} +
                                   "' must be between " +
                                   std::to_string(field.minimum_unsigned_value) + " and " +
                                   std::to_string(field.maximum_unsigned_value) + ".");
}

void assign_unsigned_value(RenderConfiguration& configuration, const std::size_t field_index,
                           const std::uint64_t value) {
    switch (field_index) {
    case 0:
        configuration.schema_version = static_cast<std::uint32_t>(value);
        break;
    case 1:
        configuration.extent.width = static_cast<std::uint32_t>(value);
        break;
    case 2:
        configuration.extent.height = static_cast<std::uint32_t>(value);
        break;
    case 3:
        configuration.samples_per_pixel = static_cast<std::uint32_t>(value);
        break;
    case 4:
        configuration.maximum_path_depth = static_cast<std::uint32_t>(value);
        break;
    case 5:
        configuration.tile_edge_length = static_cast<std::uint32_t>(value);
        break;
    case 6:
        configuration.seed = value;
        break;
    default:
        break;
    }
}

} // namespace

RenderConfigurationSchema render_configuration_schema() noexcept {
    return RenderConfigurationSchema{
        .version = CurrentRenderConfigurationSchemaVersion,
        .allows_unknown_keys = false,
        .fields = std::span<const RenderConfigurationFieldSchema>{configuration_fields},
    };
}

core::Result<RenderConfiguration>
parse_and_validate_render_configuration(const std::string_view encoded_configuration) {
    if (encoded_configuration.size() > limits::maximum_encoded_configuration_size) {
        return std::unexpected(configuration_error(
            core::StatusCode::resource_exhausted,
            "Encoded render configuration exceeds the 65536-byte safety limit."));
    }

    auto cursor = JsonCursor{encoded_configuration};
    if (!cursor.consume('{')) {
        return std::unexpected(cursor.syntax_error("Render configuration must be a JSON object."));
    }

    auto configuration = RenderConfiguration{};
    auto fields_seen = std::array<bool, configuration_fields.size()>{};
    if (!cursor.consume('}')) {
        while (true) {
            auto key = cursor.parse_string(limits::maximum_key_size, "render configuration key");
            if (!key) {
                return std::unexpected(std::move(key.error()));
            }

            const auto field_index = schema_field_index(*key);
            if (field_index == configuration_fields.size()) {
                return std::unexpected(
                    configuration_error(core::StatusCode::invalid_argument,
                                        "Unknown render configuration key '" + *key + "'."));
            }
            if (fields_seen[field_index]) {
                return std::unexpected(
                    configuration_error(core::StatusCode::invalid_argument,
                                        "Duplicate render configuration key '" + *key + "'."));
            }
            fields_seen[field_index] = true;

            if (!cursor.consume(':')) {
                return std::unexpected(cursor.syntax_error(
                    "Expected ':' after render configuration key '" + *key + "'"));
            }

            const auto& field = configuration_fields[field_index];
            if (field.value_kind == RenderConfigurationValueKind::unsigned_integer) {
                auto value = cursor.parse_unsigned_integer(field.key);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                if (field_index == 0 && *value != CurrentRenderConfigurationSchemaVersion) {
                    return std::unexpected(configuration_error(
                        core::StatusCode::incompatible,
                        "Unsupported render configuration schema version " +
                            std::to_string(*value) + "; expected " +
                            std::to_string(CurrentRenderConfigurationSchemaVersion) + "."));
                }
                if (*value < field.minimum_unsigned_value ||
                    *value > field.maximum_unsigned_value) {
                    return std::unexpected(range_error(field));
                }
                assign_unsigned_value(configuration, field_index, *value);
            } else {
                auto value =
                    cursor.parse_string(field.maximum_string_size, "XPU device identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                configuration.xpu_device_id = std::move(*value);
            }

            if (cursor.consume('}')) {
                break;
            }
            if (!cursor.consume(',')) {
                return std::unexpected(
                    cursor.syntax_error("Expected ',' or '}' after configuration value."));
            }
        }
    }

    if (!cursor.finished()) {
        return std::unexpected(
            cursor.syntax_error("Unexpected data after render configuration object."));
    }

    for (std::size_t field_index = 0; field_index < configuration_fields.size(); ++field_index) {
        const auto& field = configuration_fields[field_index];
        if (field.required && !fields_seen[field_index]) {
            return std::unexpected(configuration_error(
                core::StatusCode::invalid_argument,
                "Missing required render configuration key '" + std::string{field.key} + "'."));
        }
    }

    auto validation = validate_render_configuration(configuration);
    if (!validation) {
        return std::unexpected(std::move(validation.error()));
    }
    return configuration;
}

} // namespace blackframe::renderer
