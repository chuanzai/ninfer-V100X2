#include "ninfer/ops/host_vision.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

// The LP64 BLAS ABI is shared by the supported Linux BLAS providers. No GPU BLAS is used.
extern "C" void sgemm_(const char*, const char*, const int*, const int*, const int*,
    const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*);
// Runtime thread control so the attention body can stop BLAS from spawning workers per call.
extern "C" void openblas_set_num_threads(int);
extern "C" int openblas_get_num_threads(void);

namespace ninfer::ops {
namespace {
float half_to_float(std::uint16_t value) {
    const int exponent = (value >> 10) & 31;
    const int mantissa = value & 1023;
    float result;
    if (exponent == 0) { result = std::ldexp(static_cast<float>(mantissa), -24); }
    else if (exponent == 31) {
        result = mantissa ? std::numeric_limits<float>::quiet_NaN()
                          : std::numeric_limits<float>::infinity();
    } else { result = std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25); }
    return value & 0x8000 ? -result : result;
}

void require(bool valid, const char* message) {
    if (!valid) { throw std::invalid_argument(message); }
}

int token_count(std::size_t elements, int hidden) {
    require(hidden > 0 && elements > 0 && elements % hidden == 0 &&
            elements / hidden <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "invalid CPU Vision matrix shape");
    return static_cast<int>(elements / hidden);
}
} // namespace

float host_bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

std::uint16_t host_float_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float host_round_bf16(float value) { return host_bf16_to_float(host_float_to_bf16(value)); }

HostLinearWeight decode_host_linear(const Weight& weight) {
    const int bits = weight.qtype == QType::Q4G64_F16S ? 4 :
                     weight.qtype == QType::Q5G64_F16S ? 5 :
                     weight.qtype == QType::Q6G64_F16S ? 6 :
                     weight.qtype == QType::W8G32_F16S ? 8 : 0;
    require(bits != 0 && weight.layout == QuantLayout::RowSplit && weight.n > 0 && weight.k > 0 &&
            weight.group == (bits == 8 ? 32 : 64) && weight.padded_shape[1] >= weight.k &&
            weight.padded_shape[1] % weight.group == 0 && weight.qdata && weight.scales &&
            (bits == 4 || bits == 8 || weight.qhigh), "invalid CPU Vision row-split weight");
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* high = static_cast<const std::uint8_t*>(weight.qhigh);
    const auto* scales = static_cast<const std::byte*>(weight.scales);
    const int groups = weight.padded_shape[1] / weight.group;
    HostLinearWeight result{weight.n, weight.k, std::vector<float>(std::size_t(weight.n) * weight.k)};
    for (int row = 0; row < weight.n; ++row) {
        for (int column = 0; column < weight.k; ++column) {
            const std::size_t group = std::size_t(row) * groups + column / weight.group;
            const int element = column % weight.group;
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, scales + group * 2, 2);
            int code;
            if (bits == 8) {
                code = static_cast<std::int8_t>(codes[group * 32 + element]);
            } else {
                const auto packed = codes[group * 32 + element / 2];
                code = (packed >> ((element % 2) * 4)) & 15;
                if (bits == 5) { code |= ((high[group * 8 + element / 8] >> (element % 8)) & 1) << 4; }
                if (bits == 6) { code |= ((high[group * 16 + element / 4] >> ((element % 4) * 2)) & 3) << 4; }
                if (code & (1 << (bits - 1))) { code -= 1 << bits; }
            }
            result.values[std::size_t(row) * weight.k + column] = code * half_to_float(scale_bits);
        }
    }
    return result;
}

std::vector<float> host_linear(std::span<const float> input, const HostLinearWeight& weight) {
    const int tokens = token_count(input.size(), weight.columns);
    require(weight.rows > 0 && weight.values.size() == std::size_t(weight.rows) * weight.columns,
            "invalid CPU Vision decoded weight shape");
    std::vector<float> result(std::size_t(tokens) * weight.rows);
    const float one = 1, zero = 0;
    const char transpose = 'T', normal = 'N';
    sgemm_(&transpose, &normal, &weight.rows, &tokens, &weight.columns, &one,
           weight.values.data(), &weight.columns, input.data(), &weight.columns, &zero,
           result.data(), &weight.rows);
    const auto rounded = static_cast<std::ptrdiff_t>(result.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < rounded; ++i) {
        result[std::size_t(i)] = host_round_bf16(result[std::size_t(i)]);
    }
    return result;
}

void host_add_bias(std::vector<float>& values, std::span<const float> bias) {
    token_count(values.size(), static_cast<int>(bias.size()));
    const auto count = static_cast<std::ptrdiff_t>(values.size());
    const auto width = static_cast<std::ptrdiff_t>(bias.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        values[std::size_t(i)] = host_round_bf16(values[std::size_t(i)] + bias[std::size_t(i % width)]);
    }
}

void host_residual_add(std::vector<float>& residual, std::span<const float> addition) {
    require(residual.size() == addition.size(), "CPU Vision residual shape mismatch");
    const auto count = static_cast<std::ptrdiff_t>(residual.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        residual[std::size_t(i)] = host_round_bf16(residual[std::size_t(i)] + addition[std::size_t(i)]);
    }
}

std::vector<float> host_layer_norm(std::span<const float> input, std::span<const float> weight,
                                  std::span<const float> bias, float epsilon) {
    const int hidden = static_cast<int>(weight.size());
    const int tokens = token_count(input.size(), hidden);
    require(bias.size() == weight.size() && epsilon > 0, "CPU Vision norm shape mismatch");
    std::vector<float> result(input.size());
    #pragma omp parallel for schedule(static)
    for (int token = 0; token < tokens; ++token) {
        const auto base = std::size_t(token) * hidden;
        float sum = 0;
        for (int j = 0; j < hidden; ++j) { sum += input[base + j]; }
        const float mean = sum / hidden;
        float variance = 0;
        for (int j = 0; j < hidden; ++j) {
            const float centered = input[base + j] - mean;
            variance += centered * centered;
        }
        const float inverse = 1 / std::sqrt(variance / hidden + epsilon);
        for (int j = 0; j < hidden; ++j) {
            result[base + j] = host_round_bf16((input[base + j] - mean) * inverse * weight[j] + bias[j]);
        }
    }
    return result;
}

void host_gelu(std::vector<float>& values, bool exact) {
    const auto count = static_cast<std::ptrdiff_t>(values.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        const float value = values[std::size_t(i)];
        const float activation = exact ? 0.5F * value * (1 + std::erf(value * 0.7071067811865475F))
            : 0.5F * value * (1 + std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value)));
        values[std::size_t(i)] = host_round_bf16(activation);
    }
}

void host_vision_pos_embed(std::vector<float>& values, std::span<const float> table,
                           std::span<const std::int32_t> indices, std::span<const float> weights,
                           int hidden) {
    const int tokens = token_count(values.size(), hidden);
    require(indices.size() == std::size_t(tokens) * 4 && weights.size() == indices.size() &&
            table.size() % hidden == 0, "CPU Vision position shape mismatch");
    for (int token = 0; token < tokens; ++token) {
        for (int j = 0; j < hidden; ++j) {
            float position = 0;
            for (int corner = 0; corner < 4; ++corner) {
                const auto c = std::size_t(token) * 4 + corner;
                require(indices[c] >= 0 && std::size_t(indices[c]) < table.size() / hidden,
                        "CPU Vision position index outside table");
                position += table[std::size_t(indices[c]) * hidden + j] * weights[c];
            }
            const auto offset = std::size_t(token) * hidden + j;
            values[offset] = host_round_bf16(values[offset] + position);
        }
    }
}

void host_vision_rope(std::vector<float>& qkv, std::span<const std::int32_t> positions,
                      int heads, int head_dim) {
    require(heads == 16 && head_dim == 72, "CPU Vision requires 16 heads of dimension 72");
    const int hidden = heads * head_dim;
    const int tokens = token_count(qkv.size(), 3 * hidden);
    require(positions.size() == std::size_t(tokens) * 2, "CPU Vision RoPE position mismatch");
    for (int token = 0; token < tokens; ++token) {
        for (int pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = std::pow(10000.0F, -static_cast<float>(pair % 18) / 18);
            const float angle = positions[std::size_t(pair / 18) * tokens + token] * frequency;
            const float cosine = std::cos(angle), sine = std::sin(angle);
            for (int plane = 0; plane < 2; ++plane) {
                for (int head = 0; head < heads; ++head) {
                    const auto offset = std::size_t(token) * 3 * hidden + plane * hidden + head * head_dim + pair;
                    const float first = qkv[offset], second = qkv[offset + head_dim / 2];
                    qkv[offset] = host_round_bf16(first * cosine - second * sine);
                    qkv[offset + head_dim / 2] = host_round_bf16(second * cosine + first * sine);
                }
            }
        }
    }
}

std::vector<float> host_vision_attention(std::span<const float> qkv,
    std::span<const std::int32_t> cu_seqlens, int heads, int head_dim) {
    require(heads > 0 && head_dim > 0, "invalid CPU Vision attention geometry");
    const int hidden = heads * head_dim;
    const int tokens = token_count(qkv.size(), 3 * hidden);
    require(cu_seqlens.size() >= 2 && cu_seqlens.front() == 0 && cu_seqlens.back() == tokens,
            "CPU Vision attention segment bounds mismatch");
    std::vector<float> result(std::size_t(tokens) * hidden);
    const float scale = 1 / std::sqrt(static_cast<float>(head_dim)), one = 1, zero = 0;
    const char transpose = 'T', normal = 'N';
    // 512 query columns keeps one score tile inside L3 while still giving the kernel a wide
    // enough n to tile over: measured ~700 GFLOPS here against ~450 at 128 columns.
    constexpr int kQueryChunk = 512;
    // Every (head, chunk) pair is independent, so the whole attention body runs inside a
    // single parallel region and each thread owns its score tile. BLAS is pinned to one
    // thread inside it: at these GEMM sizes the per-call worker wakeup (sched_yield spin)
    // cost more than the math, and nesting BLAS threads under OpenMP oversubscribes badly.
    // RAII so an exception between here and the end of the pass cannot leave the process
    // with BLAS pinned to a single thread, which would quietly cost the text path 4x.
    struct BlasThreadGuard {
        int saved;
        explicit BlasThreadGuard(int threads) : saved(threads) { openblas_set_num_threads(1); }
        ~BlasThreadGuard() { openblas_set_num_threads(saved); }
        BlasThreadGuard(const BlasThreadGuard&) = delete;
        BlasThreadGuard& operator=(const BlasThreadGuard&) = delete;
    };
    const BlasThreadGuard blas_guard(openblas_get_num_threads());
    std::vector<float> planes;
    for (std::size_t segment = 1; segment < cu_seqlens.size(); ++segment) {
        const int begin = cu_seqlens[segment - 1], length = cu_seqlens[segment] - begin;
        require(length > 0 && begin >= 0 && begin + length <= tokens, "invalid CPU Vision attention segment");
        // Gather each head's q/k/v into one dense plane up front so the per-task reads stay
        // sequential instead of striding across the interleaved qkv layout.
        const auto plane = std::size_t(length) * head_dim;
        planes.assign(std::size_t(3) * heads * plane, 0.0F);
        float* q = planes.data();
        float* k = q + std::size_t(heads) * plane;
        float* v = k + std::size_t(heads) * plane;
        #pragma omp parallel for schedule(static)
        for (int head = 0; head < heads; ++head) {
            for (int i = 0; i < length; ++i) {
                const auto offset = std::size_t(begin + i) * 3 * hidden + head * head_dim;
                const auto target = std::size_t(head) * plane + std::size_t(i) * head_dim;
                std::copy_n(qkv.data() + offset, head_dim, q + target);
                std::copy_n(qkv.data() + offset + hidden, head_dim, k + target);
                std::copy_n(qkv.data() + offset + 2 * hidden, head_dim, v + target);
            }
        }
        const int chunks = (length + kQueryChunk - 1) / kQueryChunk;
        #pragma omp parallel
        {
            std::vector<float> scores(std::size_t(length) * kQueryChunk);
            std::vector<float> output(std::size_t(head_dim) * kQueryChunk);
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (int head = 0; head < heads; ++head) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int first = chunk * kQueryChunk;
                    const int count = std::min(kQueryChunk, length - first);
                    const auto base = std::size_t(head) * plane;
                    sgemm_(&transpose, &normal, &length, &count, &head_dim, &scale, k + base, &head_dim,
                           q + base + std::size_t(first) * head_dim, &head_dim, &zero, scores.data(), &length);
                    // Rows are independent. expf keeps the pass in float and lets libmvec hand
                    // the compiler a vector exp, which is what made this pass cheap to begin with.
                    for (int query = 0; query < count; ++query) {
                        float* row = scores.data() + std::size_t(query) * length;
                        const float maximum = *std::max_element(row, row + length);
                        float sum = 0;
                        #pragma omp simd reduction(+:sum)
                        for (int key = 0; key < length; ++key) { row[key] = expf(row[key] - maximum); sum += row[key]; }
                        const float inverse = 1.0F / sum;
                        #pragma omp simd
                        for (int key = 0; key < length; ++key) { row[key] *= inverse; }
                    }
                    sgemm_(&normal, &normal, &head_dim, &count, &length, &one, v + base, &head_dim,
                           scores.data(), &length, &zero, output.data(), &head_dim);
                    for (int i = 0; i < count; ++i) {
                        const auto source = std::size_t(i) * head_dim;
                        const auto target = std::size_t(begin + first + i) * hidden + head * head_dim;
                        for (int j = 0; j < head_dim; ++j) {
                            result[target + j] = host_round_bf16(output[source + j]);
                        }
                    }
                }
            }
        }
    }
    return result;
}

} // namespace ninfer::ops
