#include "SceneJsonSyntax.hpp"

#include <Blackframe/Renderer/NumericConversion.hpp>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace blackframe::engine::scene_json_syntax {
namespace {

[[nodiscard]] core::Error json_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error allocation_error(const std::string_view operation) {
    return json_error(core::StatusCode::resource_exhausted,
                      "Scene JSON " + std::string{operation} + " exhausted host memory.");
}

[[nodiscard]] core::Error length_error(const std::string_view operation) {
    return json_error(core::StatusCode::resource_exhausted,
                      "Scene JSON " + std::string{operation} + " exceeds host container limits.");
}

[[nodiscard]] bool ascii_digit(const char value) noexcept {
    return value >= '0' && value <= '9';
}

[[nodiscard]] bool json_whitespace(const char value) noexcept {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

[[nodiscard]] bool continuation_byte(const unsigned char value) noexcept {
    return value >= 0x80U && value <= 0xBFU;
}

// Returns the byte length of one shortest-form Unicode scalar, or zero for malformed UTF-8.
[[nodiscard]] std::size_t utf8_sequence_length(const std::string_view input,
                                               const std::size_t offset) noexcept {
    if (offset >= input.size()) {
        return 0U;
    }
    const auto first = static_cast<unsigned char>(input[offset]);
    if (first <= 0x7FU) {
        return 1U;
    }
    if (first >= 0xC2U && first <= 0xDFU) {
        return offset + 1U < input.size() &&
                       continuation_byte(static_cast<unsigned char>(input[offset + 1U]))
                   ? 2U
                   : 0U;
    }
    if (first >= 0xE0U && first <= 0xEFU) {
        if (offset + 2U >= input.size()) {
            return 0U;
        }
        const auto second = static_cast<unsigned char>(input[offset + 1U]);
        const auto third = static_cast<unsigned char>(input[offset + 2U]);
        const auto valid_second = first == 0xE0U   ? second >= 0xA0U && second <= 0xBFU
                                  : first == 0xEDU ? second >= 0x80U && second <= 0x9FU
                                                   : continuation_byte(second);
        return valid_second && continuation_byte(third) ? 3U : 0U;
    }
    if (first >= 0xF0U && first <= 0xF4U) {
        if (offset + 3U >= input.size()) {
            return 0U;
        }
        const auto second = static_cast<unsigned char>(input[offset + 1U]);
        const auto third = static_cast<unsigned char>(input[offset + 2U]);
        const auto fourth = static_cast<unsigned char>(input[offset + 3U]);
        const auto valid_second = first == 0xF0U   ? second >= 0x90U && second <= 0xBFU
                                  : first == 0xF4U ? second >= 0x80U && second <= 0x8FU
                                                   : continuation_byte(second);
        return valid_second && continuation_byte(third) && continuation_byte(fourth) ? 4U : 0U;
    }
    return 0U;
}

[[nodiscard]] bool valid_utf8(const std::string_view value) noexcept {
    auto offset = std::size_t{};
    while (offset < value.size()) {
        const auto length = utf8_sequence_length(value, offset);
        if (length == 0U) {
            return false;
        }
        offset += length;
    }
    return true;
}

[[nodiscard]] std::array<char, 4U> encode_utf8(const std::uint32_t code_point,
                                               std::size_t& byte_count) noexcept {
    auto bytes = std::array<char, 4U>{};
    if (code_point <= 0x7FU) {
        bytes[0] = static_cast<char>(code_point);
        byte_count = 1U;
    } else if (code_point <= 0x7FFU) {
        bytes[0] = static_cast<char>(0xC0U | (code_point >> 6U));
        bytes[1] = static_cast<char>(0x80U | (code_point & 0x3FU));
        byte_count = 2U;
    } else if (code_point <= 0xFFFFU) {
        bytes[0] = static_cast<char>(0xE0U | (code_point >> 12U));
        bytes[1] = static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU));
        bytes[2] = static_cast<char>(0x80U | (code_point & 0x3FU));
        byte_count = 3U;
    } else {
        bytes[0] = static_cast<char>(0xF0U | (code_point >> 18U));
        bytes[1] = static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU));
        bytes[2] = static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU));
        bytes[3] = static_cast<char>(0x80U | (code_point & 0x3FU));
        byte_count = 4U;
    }
    return bytes;
}

template <typename Value>
[[nodiscard]] bool invalid_limit(const Value value, const Value maximum) noexcept {
    return value == Value{} || value > maximum;
}

} // namespace

core::Status validate_limits(const SceneDescriptionJsonLimits limits) {
    const auto& maximum = MaximumSceneDescriptionJsonLimits;
    if (invalid_limit(limits.maximum_encoded_bytes, maximum.maximum_encoded_bytes) ||
        invalid_limit(limits.maximum_nesting_depth, maximum.maximum_nesting_depth) ||
        invalid_limit(limits.maximum_total_values, maximum.maximum_total_values) ||
        invalid_limit(limits.maximum_total_string_bytes, maximum.maximum_total_string_bytes) ||
        invalid_limit(limits.maximum_records_per_registry, maximum.maximum_records_per_registry) ||
        invalid_limit(limits.maximum_mesh_vertices, maximum.maximum_mesh_vertices) ||
        invalid_limit(limits.maximum_mesh_triangles, maximum.maximum_mesh_triangles) ||
        invalid_limit(limits.maximum_image_scalar_values, maximum.maximum_image_scalar_values) ||
        invalid_limit(limits.maximum_decoded_bytes, maximum.maximum_decoded_bytes)) {
        return std::unexpected(json_error(
            core::StatusCode::invalid_argument,
            "Every scene JSON limit must be non-zero and no greater than its hard maximum."));
    }
    return {};
}

Reader::Reader(const std::string_view encoded_scene, const SceneDescriptionJsonLimits limits,
               std::vector<std::uint8_t> value_slots) noexcept
    : input_{encoded_scene}, limits_{limits}, value_slots_{std::move(value_slots)} {}

core::Result<Reader> Reader::create(const std::string_view encoded_scene,
                                    const SceneDescriptionJsonLimits limits) {
    const auto limits_status = validate_limits(limits);
    if (!limits_status) {
        return std::unexpected(limits_status.error());
    }
    if (static_cast<std::uint64_t>(encoded_scene.size()) > limits.maximum_encoded_bytes) {
        return std::unexpected(json_error(core::StatusCode::resource_exhausted,
                                          "Encoded scene JSON exceeds its byte budget."));
    }
    try {
        auto slots = std::vector<std::uint8_t>{};
        slots.reserve(static_cast<std::size_t>(limits.maximum_nesting_depth) + 1U);
        slots.push_back(0U);
        return Reader{encoded_scene, limits, std::move(slots)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(allocation_error("reader creation"));
    } catch (const std::length_error&) {
        return std::unexpected(length_error("reader creation"));
    }
}

core::Error Reader::syntax_error(std::string message) const {
    message += " At byte " + std::to_string(offset_) + ".";
    return json_error(core::StatusCode::invalid_argument, std::move(message));
}

core::Error Reader::resource_error(std::string message) const {
    message += " At byte " + std::to_string(offset_) + ".";
    return json_error(core::StatusCode::resource_exhausted, std::move(message));
}

void Reader::skip_whitespace() noexcept {
    while (offset_ < input_.size() && json_whitespace(input_[offset_])) {
        ++offset_;
    }
}

bool Reader::consume(const char expected) noexcept {
    skip_whitespace();
    if (offset_ >= input_.size() || input_[offset_] != expected) {
        return false;
    }
    ++offset_;
    return true;
}

bool Reader::value_boundary() const noexcept {
    if (offset_ == input_.size()) {
        return true;
    }
    const auto value = input_[offset_];
    return json_whitespace(value) || value == ',' || value == ']' || value == '}';
}

bool Reader::consume_literal(const std::string_view literal) noexcept {
    skip_whitespace();
    if (literal.size() > input_.size() - offset_ ||
        input_.substr(offset_, literal.size()) != literal) {
        return false;
    }
    const auto end = offset_ + literal.size();
    const auto old_offset = offset_;
    offset_ = end;
    if (!value_boundary()) {
        offset_ = old_offset;
        return false;
    }
    return true;
}

core::Status Reader::claim_value() {
    if (failed_ || finished_) {
        return std::unexpected(
            syntax_error("The scene JSON reader is not available for another value"));
    }
    if (value_slots_.empty() || value_slots_.back() != 0U) {
        failed_ = true;
        return std::unexpected(
            syntax_error("A scene JSON callback must consume exactly one value"));
    }
    if (value_count_ == limits_.maximum_total_values) {
        failed_ = true;
        return std::unexpected(resource_error("Scene JSON exceeds its total-value budget"));
    }
    value_slots_.back() = 1U;
    ++value_count_;
    return {};
}

core::Status Reader::begin_container(const char opening) {
    if (auto status = claim_value(); !status) {
        return status;
    }
    if (depth_ == limits_.maximum_nesting_depth) {
        failed_ = true;
        return std::unexpected(resource_error("Scene JSON exceeds its nesting-depth budget"));
    }
    if (!consume(opening)) {
        failed_ = true;
        return std::unexpected(syntax_error(std::string{"Expected '"} + opening + "'"));
    }
    ++depth_;
    return {};
}

void Reader::end_container() noexcept {
    if (depth_ > 0U) {
        --depth_;
    }
}

core::Result<std::string> Reader::read_string_token() {
    skip_whitespace();
    if (offset_ >= input_.size() || input_[offset_] != '"') {
        return std::unexpected(syntax_error("Expected a JSON string"));
    }
    ++offset_;

    auto output = std::string{};
    const auto remaining_string_bytes = limits_.maximum_total_string_bytes - string_byte_count_;
    const auto append_decoded = [&](const std::string_view bytes) -> core::Status {
        if (static_cast<std::uint64_t>(output.size()) > remaining_string_bytes ||
            static_cast<std::uint64_t>(bytes.size()) >
                remaining_string_bytes - static_cast<std::uint64_t>(output.size())) {
            return std::unexpected(
                resource_error("Scene JSON exceeds its decoded-string byte budget"));
        }
        output.append(bytes);
        return {};
    };

    const auto parse_hex_quad = [&]() -> core::Result<std::uint32_t> {
        if (input_.size() - offset_ < 4U) {
            return std::unexpected(syntax_error("Incomplete JSON Unicode escape"));
        }
        auto value = std::uint32_t{};
        for (auto index = std::uint32_t{}; index < 4U; ++index) {
            const auto character = input_[offset_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') {
                value |= static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                value |= static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                value |= static_cast<std::uint32_t>(character - 'A' + 10);
            } else {
                return std::unexpected(syntax_error("Invalid JSON Unicode escape"));
            }
        }
        return value;
    };

    while (offset_ < input_.size()) {
        const auto character_offset = offset_;
        const auto character = static_cast<unsigned char>(input_[offset_++]);
        if (character == static_cast<unsigned char>('"')) {
            string_byte_count_ += static_cast<std::uint64_t>(output.size());
            return output;
        }
        if (character < 0x20U) {
            return std::unexpected(syntax_error("JSON strings cannot contain raw control bytes"));
        }
        if (character == static_cast<unsigned char>('\\')) {
            if (offset_ >= input_.size()) {
                return std::unexpected(syntax_error("Unterminated JSON escape sequence"));
            }
            const auto escape = input_[offset_++];
            switch (escape) {
            case '"':
            case '\\':
            case '/': {
                const auto byte = std::array{escape};
                if (auto status = append_decoded(std::string_view{byte.data(), byte.size()});
                    !status) {
                    return std::unexpected(status.error());
                }
                break;
            }
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't': {
                const auto decoded = escape == 'b'   ? '\b'
                                     : escape == 'f' ? '\f'
                                     : escape == 'n' ? '\n'
                                     : escape == 'r' ? '\r'
                                                     : '\t';
                const auto byte = std::array{decoded};
                if (auto status = append_decoded(std::string_view{byte.data(), byte.size()});
                    !status) {
                    return std::unexpected(status.error());
                }
                break;
            }
            case 'u': {
                auto first = parse_hex_quad();
                if (!first) {
                    return std::unexpected(first.error());
                }
                if (*first >= 0xDC00U && *first <= 0xDFFFU) {
                    return std::unexpected(syntax_error("Unexpected low Unicode surrogate"));
                }
                auto code_point = *first;
                if (*first >= 0xD800U && *first <= 0xDBFFU) {
                    if (input_.size() - offset_ < 2U || input_[offset_] != '\\' ||
                        input_[offset_ + 1U] != 'u') {
                        return std::unexpected(syntax_error("Missing low Unicode surrogate"));
                    }
                    offset_ += 2U;
                    auto second = parse_hex_quad();
                    if (!second) {
                        return std::unexpected(second.error());
                    }
                    if (*second < 0xDC00U || *second > 0xDFFFU) {
                        return std::unexpected(syntax_error("Invalid low Unicode surrogate"));
                    }
                    code_point = 0x10000U + ((*first - 0xD800U) << 10U) + (*second - 0xDC00U);
                }
                auto byte_count = std::size_t{};
                const auto bytes = encode_utf8(code_point, byte_count);
                if (auto status = append_decoded(std::string_view{bytes.data(), byte_count});
                    !status) {
                    return std::unexpected(status.error());
                }
                break;
            }
            default:
                return std::unexpected(syntax_error("Unknown JSON escape sequence"));
            }
            continue;
        }
        if (character <= 0x7FU) {
            const auto byte = std::array{static_cast<char>(character)};
            if (auto status = append_decoded(std::string_view{byte.data(), byte.size()}); !status) {
                return std::unexpected(status.error());
            }
            continue;
        }

        const auto sequence_length = utf8_sequence_length(input_, character_offset);
        if (sequence_length == 0U) {
            return std::unexpected(syntax_error("JSON string contains malformed UTF-8"));
        }
        offset_ = character_offset + sequence_length;
        if (auto status = append_decoded(input_.substr(character_offset, sequence_length));
            !status) {
            return std::unexpected(status.error());
        }
    }
    return std::unexpected(syntax_error("Unterminated JSON string"));
}

core::Result<std::string_view> Reader::read_number_token(const bool integer_only,
                                                         const bool unsigned_only) {
    skip_whitespace();
    const auto start = offset_;
    if (offset_ < input_.size() && input_[offset_] == '-') {
        if (unsigned_only) {
            return std::unexpected(syntax_error("Expected an unsigned JSON integer"));
        }
        ++offset_;
    }
    if (offset_ >= input_.size() || !ascii_digit(input_[offset_])) {
        return std::unexpected(
            syntax_error(integer_only ? "Expected a JSON integer" : "Expected a JSON number"));
    }
    if (input_[offset_] == '0') {
        ++offset_;
        if (offset_ < input_.size() && ascii_digit(input_[offset_])) {
            return std::unexpected(syntax_error("JSON numbers cannot contain leading zeroes"));
        }
    } else {
        while (offset_ < input_.size() && ascii_digit(input_[offset_])) {
            ++offset_;
        }
    }
    if (offset_ < input_.size() && input_[offset_] == '.') {
        if (integer_only) {
            return std::unexpected(syntax_error("Expected an integer without a fraction"));
        }
        ++offset_;
        const auto fraction_start = offset_;
        while (offset_ < input_.size() && ascii_digit(input_[offset_])) {
            ++offset_;
        }
        if (offset_ == fraction_start) {
            return std::unexpected(syntax_error("A JSON fraction requires at least one digit"));
        }
    }
    if (offset_ < input_.size() && (input_[offset_] == 'e' || input_[offset_] == 'E')) {
        if (integer_only) {
            return std::unexpected(syntax_error("Expected an integer without an exponent"));
        }
        ++offset_;
        if (offset_ < input_.size() && (input_[offset_] == '+' || input_[offset_] == '-')) {
            ++offset_;
        }
        const auto exponent_start = offset_;
        while (offset_ < input_.size() && ascii_digit(input_[offset_])) {
            ++offset_;
        }
        if (offset_ == exponent_start) {
            return std::unexpected(syntax_error("A JSON exponent requires at least one digit"));
        }
    }
    if (!value_boundary()) {
        return std::unexpected(syntax_error("Invalid byte after JSON number"));
    }
    return input_.substr(start, offset_ - start);
}

core::Status Reader::read_object_impl(void* const context, const ObjectCallback callback) {
    try {
        if (callback == nullptr) {
            failed_ = true;
            return std::unexpected(syntax_error("A scene JSON object callback is required"));
        }
        if (auto status = begin_container('{'); !status) {
            return status;
        }
        if (consume('}')) {
            end_container();
            return {};
        }

        auto keys = std::set<std::string, std::less<>>{};
        while (true) {
            auto key_value = read_string_token();
            if (!key_value) {
                failed_ = true;
                end_container();
                return std::unexpected(key_value.error());
            }
            auto [key, inserted] = keys.insert(std::move(*key_value));
            if (!inserted) {
                failed_ = true;
                end_container();
                return std::unexpected(
                    syntax_error("Duplicate decoded JSON object key '" + *key + "'"));
            }
            if (!consume(':')) {
                failed_ = true;
                end_container();
                return std::unexpected(syntax_error("Expected ':' after JSON object key"));
            }

            value_slots_.push_back(0U);
            auto status = callback(context, *key);
            const auto consumed = value_slots_.back() == 1U;
            value_slots_.pop_back();
            if (!status) {
                failed_ = true;
                end_container();
                return status;
            }
            if (!consumed) {
                failed_ = true;
                end_container();
                return std::unexpected(
                    json_error(core::StatusCode::internal_error,
                               "A scene JSON object callback did not consume exactly one value."));
            }

            if (consume('}')) {
                end_container();
                return {};
            }
            if (!consume(',')) {
                failed_ = true;
                end_container();
                return std::unexpected(syntax_error("Expected ',' or '}' after JSON object value"));
            }
        }
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("object parsing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("object parsing"));
    }
}

core::Status Reader::read_array_impl(void* const context, const ArrayCallback callback) {
    try {
        if (callback == nullptr) {
            failed_ = true;
            return std::unexpected(syntax_error("A scene JSON array callback is required"));
        }
        if (auto status = begin_container('['); !status) {
            return status;
        }
        if (consume(']')) {
            end_container();
            return {};
        }

        auto index = std::uint64_t{};
        while (true) {
            value_slots_.push_back(0U);
            auto status = callback(context, index);
            const auto consumed = value_slots_.back() == 1U;
            value_slots_.pop_back();
            if (!status) {
                failed_ = true;
                end_container();
                return status;
            }
            if (!consumed) {
                failed_ = true;
                end_container();
                return std::unexpected(
                    json_error(core::StatusCode::internal_error,
                               "A scene JSON array callback did not consume exactly one value."));
            }
            if (consume(']')) {
                end_container();
                return {};
            }
            if (!consume(',')) {
                failed_ = true;
                end_container();
                return std::unexpected(syntax_error("Expected ',' or ']' after JSON array value"));
            }
            if (index == std::numeric_limits<std::uint64_t>::max()) {
                failed_ = true;
                end_container();
                return std::unexpected(resource_error("JSON array index is not representable"));
            }
            ++index;
        }
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("array parsing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("array parsing"));
    }
}

core::Result<std::string> Reader::read_string() {
    try {
        if (auto status = claim_value(); !status) {
            return std::unexpected(status.error());
        }
        auto result = read_string_token();
        if (!result) {
            failed_ = true;
        }
        return result;
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("string parsing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("string parsing"));
    }
}

core::Result<std::uint64_t> Reader::read_u64() {
    if (auto status = claim_value(); !status) {
        return std::unexpected(status.error());
    }
    const auto token = read_number_token(true, true);
    if (!token) {
        failed_ = true;
        return std::unexpected(token.error());
    }
    auto value = std::uint64_t{};
    const auto parsed = std::from_chars(token->data(), token->data() + token->size(), value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != token->data() + token->size()) {
        failed_ = true;
        return std::unexpected(syntax_error("Unsigned JSON integer is outside uint64 range"));
    }
    return value;
}

core::Result<std::int64_t> Reader::read_i64() {
    if (auto status = claim_value(); !status) {
        return std::unexpected(status.error());
    }
    const auto token = read_number_token(true, false);
    if (!token) {
        failed_ = true;
        return std::unexpected(token.error());
    }
    auto value = std::int64_t{};
    const auto parsed = std::from_chars(token->data(), token->data() + token->size(), value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != token->data() + token->size()) {
        failed_ = true;
        return std::unexpected(syntax_error("Signed JSON integer is outside int64 range"));
    }
    return value;
}

core::Result<float> Reader::read_float() {
    if (auto status = claim_value(); !status) {
        return std::unexpected(status.error());
    }
    const auto token = read_number_token(false, false);
    if (!token) {
        failed_ = true;
        return std::unexpected(token.error());
    }
    auto reference = renderer::ReferenceScalar{};
    const auto parsed = std::from_chars(token->data(), token->data() + token->size(), reference,
                                        std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != token->data() + token->size() ||
        !std::isfinite(reference)) {
        failed_ = true;
        return std::unexpected(syntax_error("JSON number is not a finite representable scalar"));
    }
    const auto transport = renderer::to_transport_scalar(reference);
    if (!transport) {
        failed_ = true;
        return std::unexpected(transport.error());
    }
    return *transport;
}

core::Status Reader::read_null() {
    if (auto status = claim_value(); !status) {
        return status;
    }
    if (!consume_literal("null")) {
        failed_ = true;
        return std::unexpected(syntax_error("Expected JSON null"));
    }
    return {};
}

bool Reader::peek_null() noexcept {
    if (failed_ || finished_) {
        return false;
    }
    skip_whitespace();
    if (input_.size() - offset_ < 4U || input_.substr(offset_, 4U) != "null") {
        return false;
    }
    const auto end = offset_ + 4U;
    return end == input_.size() || json_whitespace(input_[end]) || input_[end] == ',' ||
           input_[end] == ']' || input_[end] == '}';
}

bool Reader::peek_array() noexcept {
    if (failed_ || finished_) {
        return false;
    }
    skip_whitespace();
    return offset_ < input_.size() && input_[offset_] == '[';
}

core::Status Reader::finish() {
    if (failed_ || finished_ || depth_ != 0U || value_slots_.size() != 1U ||
        value_slots_.front() != 1U) {
        failed_ = true;
        return std::unexpected(
            syntax_error("Scene JSON must contain exactly one complete root value"));
    }
    skip_whitespace();
    if (offset_ != input_.size()) {
        failed_ = true;
        return std::unexpected(syntax_error("Unexpected data after scene JSON root value"));
    }
    finished_ = true;
    return {};
}

Writer::Writer(const SceneDescriptionJsonLimits limits) noexcept : limits_{limits} {}

core::Result<Writer> Writer::create(const SceneDescriptionJsonLimits limits) {
    const auto status = validate_limits(limits);
    if (!status) {
        return std::unexpected(status.error());
    }
    try {
        auto writer = Writer{limits};
        writer.stack_.reserve(limits.maximum_nesting_depth);
        return writer;
    } catch (const std::bad_alloc&) {
        return std::unexpected(allocation_error("writer creation"));
    } catch (const std::length_error&) {
        return std::unexpected(length_error("writer creation"));
    }
}

core::Error Writer::state_error(std::string message) const {
    return json_error(core::StatusCode::invalid_argument, std::move(message));
}

core::Error Writer::resource_error(std::string message) const {
    return json_error(core::StatusCode::resource_exhausted, std::move(message));
}

core::Status Writer::append(const std::string_view bytes) {
    if (static_cast<std::uint64_t>(output_.size()) > limits_.maximum_encoded_bytes ||
        static_cast<std::uint64_t>(bytes.size()) >
            limits_.maximum_encoded_bytes - static_cast<std::uint64_t>(output_.size())) {
        failed_ = true;
        return std::unexpected(resource_error("Canonical scene JSON exceeds its output budget."));
    }
    output_.append(bytes);
    return {};
}

core::Status Writer::append_character(const char value) {
    return append(std::string_view{&value, 1U});
}

core::Status Writer::count_value() {
    if (value_count_ == limits_.maximum_total_values) {
        failed_ = true;
        return std::unexpected(
            resource_error("Canonical scene JSON exceeds its total-value budget."));
    }
    ++value_count_;
    return {};
}

core::Status Writer::count_string(const std::size_t byte_count) {
    if (string_byte_count_ > limits_.maximum_total_string_bytes ||
        static_cast<std::uint64_t>(byte_count) >
            limits_.maximum_total_string_bytes - string_byte_count_) {
        failed_ = true;
        return std::unexpected(
            resource_error("Canonical scene JSON exceeds its decoded-string byte budget."));
    }
    string_byte_count_ += static_cast<std::uint64_t>(byte_count);
    return {};
}

core::Status Writer::begin_value() {
    if (failed_ || finished_) {
        return std::unexpected(state_error("The scene JSON writer cannot accept another value."));
    }
    if (auto status = count_value(); !status) {
        return status;
    }
    if (stack_.empty()) {
        if (root_written_) {
            failed_ = true;
            return std::unexpected(
                state_error("Canonical scene JSON can contain only one root value."));
        }
        root_written_ = true;
        return {};
    }

    auto& parent = stack_.back();
    if (parent.kind == ContainerKind::object) {
        if (!parent.awaiting_value) {
            failed_ = true;
            return std::unexpected(
                state_error("A canonical JSON object value requires a preceding key."));
        }
        parent.awaiting_value = false;
        ++parent.element_count;
        return {};
    }
    if (parent.element_count > 0U) {
        if (auto status = append_character(','); !status) {
            return status;
        }
    }
    ++parent.element_count;
    return {};
}

core::Status Writer::begin_container(const ContainerKind kind, const char opening) {
    try {
        if (auto status = begin_value(); !status) {
            return status;
        }
        if (stack_.size() == limits_.maximum_nesting_depth) {
            failed_ = true;
            return std::unexpected(
                resource_error("Canonical scene JSON exceeds its nesting-depth budget."));
        }
        if (auto status = append_character(opening); !status) {
            return status;
        }
        stack_.push_back(Frame{.kind = kind});
        return {};
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("container writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("container writing"));
    }
}

core::Status Writer::end_container(const ContainerKind kind, const char closing) {
    try {
        if (failed_ || finished_ || stack_.empty() || stack_.back().kind != kind) {
            failed_ = true;
            return std::unexpected(state_error("Canonical JSON containers must be balanced."));
        }
        if (kind == ContainerKind::object && stack_.back().awaiting_value) {
            failed_ = true;
            return std::unexpected(
                state_error("A canonical JSON object key is missing its value."));
        }
        if (auto status = append_character(closing); !status) {
            return status;
        }
        stack_.pop_back();
        return {};
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("container completion"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("container completion"));
    }
}

core::Status Writer::begin_object() {
    return begin_container(ContainerKind::object, '{');
}

core::Status Writer::end_object() {
    return end_container(ContainerKind::object, '}');
}

core::Status Writer::begin_array() {
    return begin_container(ContainerKind::array, '[');
}

core::Status Writer::end_array() {
    return end_container(ContainerKind::array, ']');
}

core::Status Writer::append_escaped_string(const std::string_view value,
                                           const bool count_string_bytes) {
    if (!valid_utf8(value)) {
        failed_ = true;
        return std::unexpected(
            state_error("Canonical scene JSON strings must contain valid UTF-8."));
    }
    if (count_string_bytes) {
        if (auto status = count_string(value.size()); !status) {
            return status;
        }
    }
    if (auto status = append_character('"'); !status) {
        return status;
    }
    constexpr auto hex = std::string_view{"0123456789abcdef"};
    auto offset = std::size_t{};
    while (offset < value.size()) {
        const auto character = static_cast<unsigned char>(value[offset]);
        if (character > 0x7FU) {
            const auto length = utf8_sequence_length(value, offset);
            if (auto status = append(value.substr(offset, length)); !status) {
                return status;
            }
            offset += length;
            continue;
        }
        ++offset;
        switch (character) {
        case '"':
            if (auto status = append("\\\""); !status) {
                return status;
            }
            break;
        case '\\':
            if (auto status = append("\\\\"); !status) {
                return status;
            }
            break;
        case '\b':
            if (auto status = append("\\b"); !status) {
                return status;
            }
            break;
        case '\f':
            if (auto status = append("\\f"); !status) {
                return status;
            }
            break;
        case '\n':
            if (auto status = append("\\n"); !status) {
                return status;
            }
            break;
        case '\r':
            if (auto status = append("\\r"); !status) {
                return status;
            }
            break;
        case '\t':
            if (auto status = append("\\t"); !status) {
                return status;
            }
            break;
        default:
            if (character < 0x20U) {
                const auto escaped = std::array{
                    '\\', 'u', '0', '0', hex[(character >> 4U) & 0xFU], hex[character & 0xFU]};
                if (auto status = append(std::string_view{escaped.data(), escaped.size()});
                    !status) {
                    return status;
                }
            } else {
                const auto byte = static_cast<char>(character);
                if (auto status = append_character(byte); !status) {
                    return status;
                }
            }
            break;
        }
    }
    return append_character('"');
}

core::Status Writer::key(const std::string_view value) {
    try {
        if (failed_ || finished_ || stack_.empty() || stack_.back().kind != ContainerKind::object ||
            stack_.back().awaiting_value) {
            failed_ = true;
            return std::unexpected(
                state_error("A canonical JSON key requires an object awaiting its next key."));
        }
        if (!valid_utf8(value)) {
            failed_ = true;
            return std::unexpected(
                state_error("Canonical scene JSON keys must contain valid UTF-8."));
        }
        auto& frame = stack_.back();
        const auto [unused, inserted] = frame.keys.emplace(value);
        static_cast<void>(unused);
        if (!inserted) {
            failed_ = true;
            return std::unexpected(state_error("Canonical scene JSON cannot contain duplicate "
                                               "object keys."));
        }
        if (auto status = count_string(value.size()); !status) {
            return status;
        }
        if (frame.element_count > 0U) {
            if (auto status = append_character(','); !status) {
                return status;
            }
        }
        if (auto status = append_escaped_string(value, false); !status) {
            return status;
        }
        if (auto status = append_character(':'); !status) {
            return status;
        }
        frame.awaiting_value = true;
        return {};
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("object-key writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("object-key writing"));
    }
}

core::Status Writer::write_null() {
    try {
        if (auto status = begin_value(); !status) {
            return status;
        }
        return append("null");
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("null writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("null writing"));
    }
}

core::Status Writer::write_string(const std::string_view value) {
    try {
        if (auto status = begin_value(); !status) {
            return status;
        }
        return append_escaped_string(value, true);
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("string writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("string writing"));
    }
}

core::Status Writer::write_u64(const std::uint64_t value) {
    try {
        if (auto status = begin_value(); !status) {
            return status;
        }
        auto buffer = std::array<char, 32U>{};
        const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        if (converted.ec != std::errc{}) {
            failed_ = true;
            return std::unexpected(state_error("A uint64 JSON value cannot be formatted."));
        }
        return append(std::string_view{buffer.data(),
                                       static_cast<std::size_t>(converted.ptr - buffer.data())});
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("uint64 writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("uint64 writing"));
    }
}

core::Status Writer::write_i64(const std::int64_t value) {
    try {
        if (auto status = begin_value(); !status) {
            return status;
        }
        auto buffer = std::array<char, 32U>{};
        const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        if (converted.ec != std::errc{}) {
            failed_ = true;
            return std::unexpected(state_error("An int64 JSON value cannot be formatted."));
        }
        return append(std::string_view{buffer.data(),
                                       static_cast<std::size_t>(converted.ptr - buffer.data())});
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("int64 writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("int64 writing"));
    }
}

core::Status Writer::write_float(const float value) {
    try {
        if (!std::isfinite(value)) {
            failed_ = true;
            return std::unexpected(state_error("Canonical scene JSON floats must be finite."));
        }
        if (auto status = begin_value(); !status) {
            return status;
        }
        auto buffer = std::array<char, 64U>{};
        const auto converted =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                          std::chars_format::general, std::numeric_limits<float>::max_digits10);
        if (converted.ec != std::errc{}) {
            failed_ = true;
            return std::unexpected(state_error("A float JSON value cannot be formatted."));
        }
        return append(std::string_view{buffer.data(),
                                       static_cast<std::size_t>(converted.ptr - buffer.data())});
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("float writing"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("float writing"));
    }
}

core::Result<std::string> Writer::finish() {
    try {
        if (failed_ || finished_ || !root_written_ || !stack_.empty()) {
            failed_ = true;
            return std::unexpected(
                state_error("Canonical scene JSON requires one complete balanced root value."));
        }
        finished_ = true;
        return std::move(output_);
    } catch (const std::bad_alloc&) {
        failed_ = true;
        return std::unexpected(allocation_error("completion"));
    } catch (const std::length_error&) {
        failed_ = true;
        return std::unexpected(length_error("completion"));
    }
}

} // namespace blackframe::engine::scene_json_syntax
