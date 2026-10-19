/*
 * The VU's FMAC arithmetic on arm64 without touching FPCR: the lanes worked
 * in the processor's own mode (to nearest, denormals kept) and each result
 * then taken one step toward zero where the exact one lies there - which is
 * what rounding toward zero gives. For VU0's macro mode (vu0.cpp's Cop2),
 * which ran each instruction inside an FPCR switch and back (fp_mode.h): on
 * Apple's cores the two writes cost ~22 ns, six times the arithmetic, and
 * the race's EE code issues tens of thousands an frame (19 Oct, an A12:
 * lifted::Cop2 a fifth of the EE thread).
 *
 * The same numbers as LanesSse<Op, 0, false> under fp_mode.h's VU mode:
 *  - operands cleaned (CleanLanes): no infinities, NaNs or denormals in;
 *  - a + b, a - b: the rounded sum s and its exact error e (Fast2Sum, the
 *    larger first: exact through gradual underflow, and no overflow on the
 *    way); a * b: e = fma(a, b, -s), exact (scaled up where it would fall
 *    below the denormals). Where e is non-zero and of the other sign than s, the exact
 *    result lies between s and zero: one step toward zero (the bits less
 *    one: a float's magnitude is monotonic in its bits) is the toward-zero
 *    result. An overflow (s infinite) cleans to the largest finite either way.
 *  - results cleaned (CleanLanes): what FZ would have flushed is flushed.
 * VuRtzIsTheMode (host tests, arm64) compares the two on random operands.
 */

#ifndef PS2RT_VU_RTZ_H
#define PS2RT_VU_RTZ_H

#if (defined(__aarch64__) || defined(__arm64__)) && !defined(_MSC_VER)
#include <arm_neon.h>
#include <cstdint>

#define PS2RT_VU_RTZ 1

namespace lifted {
namespace rtz {

inline uint32x4_t Clean(uint32x4_t v) {
    const uint32x4_t magnitude = vandq_u32(v, vdupq_n_u32(0x7FFFFFFFu));
    const uint32x4_t sign = veorq_u32(v, magnitude);
    const uint32x4_t clamped = vminq_u32(magnitude, vdupq_n_u32(0x7F7FFFFFu));
    const uint32x4_t denormal = vcltq_u32(magnitude, vdupq_n_u32(0x00800000u));
    return vorrq_u32(sign, vbicq_u32(clamped, denormal));
}

inline float32x4_t F(uint32x4_t v) { return vreinterpretq_f32_u32(v); }
inline uint32x4_t U(float32x4_t v) { return vreinterpretq_u32_f32(v); }

// s, rounded to nearest, and e, the exact result less s: toward zero.
inline uint32x4_t TowardZero(float32x4_t s, float32x4_t e) {
    const uint32x4_t sb = U(s), eb = U(e);
    const uint32x4_t opposite = vtstq_u32(veorq_u32(sb, eb), vdupq_n_u32(0x80000000u));
    const uint32x4_t inexact = vtstq_u32(eb, vdupq_n_u32(0x7FFFFFFFu));
    const uint32x4_t nonzero = vtstq_u32(sb, vdupq_n_u32(0x7FFFFFFFu));
    const uint32x4_t step = vandq_u32(vandq_u32(opposite, inexact), nonzero);
    return vaddq_u32(sb, step);         // step is all ones (-1) where it applies
}

// Fast2Sum, the larger operand first: s - big is exact and never overflows
// (TwoSum's s - a did, near the largest finite).
inline uint32x4_t Add(float32x4_t a, float32x4_t b) {
    const uint32x4_t a_larger = vcageq_f32(a, b);
    const float32x4_t big = vbslq_f32(a_larger, a, b), small = vbslq_f32(a_larger, b, a);
    const float32x4_t s = vaddq_f32(big, small);
    const float32x4_t e = vsubq_f32(small, vsubq_f32(s, big));
    return TowardZero(s, e);
}

// e = fma(a, b, -s): exact unless it falls below the denormals - a product
// within 2^24 of the smallest normal - so there a and b are taken 2^32
// times larger each (exact), and s 2^64 times, for e alone.
inline uint32x4_t Mul(float32x4_t a, float32x4_t b) {
    const float32x4_t s = vmulq_f32(a, b);
    const uint32x4_t tiny = vcaltq_f32(s, vdupq_n_f32(0x1p-60f));
    const float32x4_t up = vbslq_f32(tiny, vdupq_n_f32(0x1p32f), vdupq_n_f32(1.0f));
    const float32x4_t up2 = vmulq_f32(up, up);
    const float32x4_t e = vfmaq_f32(vnegq_f32(vmulq_f32(s, up2)), vmulq_f32(a, up), vmulq_f32(b, up));
    return TowardZero(s, e);
}

enum { kAdd, kSub, kMul, kMadd, kMsub };

/* LanesSse<Op, 0, false>'s result for kAdd ... kMsub (vu_math.h's order). */
template <int Op>
inline uint32x4_t Lanes(uint32x4_t a, uint32x4_t b, uint32x4_t acc) {
    a = Clean(a);
    b = Clean(b);
    if constexpr (Op == kAdd) return Clean(Add(F(a), F(b)));
    else if constexpr (Op == kSub) return Clean(Add(F(a), vnegq_f32(F(b))));
    else if constexpr (Op == kMul) return Clean(Mul(F(a), F(b)));
    else {
        const float32x4_t product = F(Clean(Mul(F(a), F(b))));
        const float32x4_t total = F(Clean(acc));
        return Clean(Add(total, Op == kMadd ? product : vnegq_f32(product)));
    }
}

}  // namespace rtz
}  // namespace lifted

#endif
#endif
