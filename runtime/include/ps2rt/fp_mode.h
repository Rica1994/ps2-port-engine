/*
 * The VUs' float arithmetic as the console's: rounded toward zero, denormals
 * flushed (inputs and results). The host's floating-point control register set
 * so while a VU instruction or microprogram runs, and put back after:
 *   - x86-64: MXCSR - RC toward zero (0x6000), FTZ (0x8000), DAZ (0x0040);
 *   - arm64:  FPCR  - RMode toward zero (bits 22-23 = 3), FZ (bit 24), which on
 *             AArch64 flushes denormal inputs and results of scalar and
 *             Advanced SIMD operations alike.
 */

#ifndef PS2RT_FP_MODE_H
#define PS2RT_FP_MODE_H

#include <cstdint>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define PS2RT_FP_X86 1
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define PS2RT_FP_ARM64 1
#endif

namespace lifted {

#if defined(PS2RT_FP_ARM64) && !defined(_MSC_VER)
inline std::uint64_t ReadFpcr() {
    std::uint64_t value;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
    return value;
}
inline void WriteFpcr(std::uint64_t value) { __asm__ __volatile__("msr fpcr, %0" : : "r"(value) : "memory"); }
#endif

/* While it lives: VU arithmetic (toward zero, denormals flushed); `on` false
   leaves the mode as it is. */
class VuFloatMode {
public:
    explicit VuFloatMode(bool on = true) : on_(on) {
        if (!on_) return;
#if defined(PS2RT_FP_X86)
        saved_ = _mm_getcsr();
        _mm_setcsr((unsigned(saved_) & ~0x6000u) | 0x6000u | 0x8000u | 0x0040u);
#elif defined(PS2RT_FP_ARM64) && !defined(_MSC_VER)
        saved_ = ReadFpcr();
        WriteFpcr(saved_ | (3ull << 22) | (1ull << 24));
#endif
    }
    ~VuFloatMode() {
        if (!on_) return;
#if defined(PS2RT_FP_X86)
        _mm_setcsr(unsigned(saved_));
#elif defined(PS2RT_FP_ARM64) && !defined(_MSC_VER)
        WriteFpcr(saved_);
#endif
    }
    VuFloatMode(const VuFloatMode&) = delete;
    VuFloatMode& operator=(const VuFloatMode&) = delete;

private:
    bool on_;
    std::uint64_t saved_ = 0;
};

}  // namespace lifted

#endif
