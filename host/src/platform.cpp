/*
 * The host's operating-system calls (platform.h): Windows' own where the PC
 * port has always used them, POSIX and SDL3 on Linux, macOS, iOS/iPadOS and
 * Android.
 *
 * Fibers outside Windows are switched by hand: the callee-saved registers
 * pushed on the running stack, the stack pointer swapped, the other's popped -
 * for arm64 (Apple silicon, iPads, Android) and x86-64 (Linux, Intel Macs). No
 * ucontext: iOS and Android's C libraries do not have a working one, and a
 * swap that saves no signal mask is a fraction of its cost.
 */

#include "ps2host/platform.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "synchronization.lib")
#pragma comment(lib, "winmm.lib")
#else
#include <pthread.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread/qos.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#endif

#ifdef PS2HOST_VULKAN
#include <SDL3/SDL_messagebox.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_system.h>
#endif

namespace ps2host {
namespace platform {

/* ======================================================================== */
/* Fibers                                                                    */
/* ======================================================================== */

#ifdef _WIN32

Fiber FiberFromThread() {
    Fiber f = ConvertThreadToFiber(nullptr);
    return f ? f : GetCurrentFiber();
}
Fiber FiberCreate(std::size_t stack_bytes, FiberEntry entry, void* parameter) {
    return CreateFiber(stack_bytes, reinterpret_cast<LPFIBER_START_ROUTINE>(entry), parameter);
}
void FiberSwitch(Fiber to) { SwitchToFiber(to); }
void FiberDelete(Fiber fiber) { DeleteFiber(fiber); }

#else

#if defined(__APPLE__)
#define PS2_ASM_NAME(name) "_" #name
#else
#define PS2_ASM_NAME(name) #name
#endif

extern "C" void ps2_fiber_swap(void** save_sp, void* load_sp);
extern "C" void ps2_fiber_start();

#if defined(__aarch64__) || defined(__arm64__)
/* x19-x28, the frame pointer and the link register, d8-d15: AAPCS64's
   callee-saved set (x18 is the platform's, left alone). 176 bytes, 16-aligned. */
__asm__(
    ".text\n"
    ".p2align 2\n"
    ".globl " PS2_ASM_NAME(ps2_fiber_swap) "\n"
    PS2_ASM_NAME(ps2_fiber_swap) ":\n"
    "    sub sp, sp, #176\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    stp d8, d9, [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mov x9, sp\n"
    "    str x9, [x0]\n"
    "    mov sp, x1\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    ldp d8, d9, [sp, #96]\n"
    "    ldp d10, d11, [sp, #112]\n"
    "    ldp d12, d13, [sp, #128]\n"
    "    ldp d14, d15, [sp, #144]\n"
    "    add sp, sp, #176\n"
    "    ret\n"
    ".p2align 2\n"
    ".globl " PS2_ASM_NAME(ps2_fiber_start) "\n"
    PS2_ASM_NAME(ps2_fiber_start) ":\n"
    "    mov x0, x20\n"
    "    blr x19\n"
    "    brk #0\n");
constexpr std::size_t kSwapFrame = 176;
#elif defined(__x86_64__)
/* rbp, rbx, r12-r15, and the SSE and x87 control words: System V's
   callee-saved set. */
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl " PS2_ASM_NAME(ps2_fiber_swap) "\n"
    PS2_ASM_NAME(ps2_fiber_swap) ":\n"
    "    pushq %rbp\n"
    "    pushq %rbx\n"
    "    pushq %r12\n"
    "    pushq %r13\n"
    "    pushq %r14\n"
    "    pushq %r15\n"
    "    subq $8, %rsp\n"
    "    stmxcsr (%rsp)\n"
    "    fnstcw 4(%rsp)\n"
    "    movq %rsp, (%rdi)\n"
    "    movq %rsi, %rsp\n"
    "    ldmxcsr (%rsp)\n"
    "    fldcw 4(%rsp)\n"
    "    addq $8, %rsp\n"
    "    popq %r15\n"
    "    popq %r14\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rbx\n"
    "    popq %rbp\n"
    "    ret\n"
    ".p2align 4\n"
    ".globl " PS2_ASM_NAME(ps2_fiber_start) "\n"
    PS2_ASM_NAME(ps2_fiber_start) ":\n"
    "    movq %r13, %rdi\n"
    "    callq *%r12\n"
    "    ud2\n");
#else
#error "ps2host fibers: arm64 or x86-64 only"
#endif

namespace {

struct FiberImpl {
    void* sp = nullptr;             // saved while it is not running
    void* memory = nullptr;         // its stack's mapping (null: a thread's own)
    std::size_t mapped = 0;
};

thread_local FiberImpl* t_current = nullptr;

std::size_t PageSize() {
    static const std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
    return page;
}

}  // namespace

Fiber FiberFromThread() {
    if (!t_current) t_current = new FiberImpl;
    return t_current;
}

Fiber FiberCreate(std::size_t stack_bytes, FiberEntry entry, void* parameter) {
    const std::size_t page = PageSize();
    const std::size_t size = (std::max<std::size_t>(stack_bytes, 64 * 1024) + page - 1) / page * page;
    // A guard page below the stack: an overflow faults, not corrupts.
    void* memory = mmap(nullptr, size + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) return nullptr;
    mprotect(memory, page, PROT_NONE);
    auto* f = new FiberImpl;
    f->memory = memory;
    f->mapped = size + page;
    auto top = reinterpret_cast<std::uintptr_t>(memory) + size + page;
    top &= ~std::uintptr_t(15);
#if defined(__aarch64__) || defined(__arm64__)
    auto* frame = reinterpret_cast<std::uint64_t*>(top - kSwapFrame);
    std::memset(frame, 0, kSwapFrame);
    frame[0] = reinterpret_cast<std::uint64_t>(entry);          // x19
    frame[1] = reinterpret_cast<std::uint64_t>(parameter);      // x20
    frame[10] = 0;                                              // x29: the end of the frame chain
    frame[11] = reinterpret_cast<std::uint64_t>(&ps2_fiber_start);   // x30: where the swap returns
    f->sp = frame;
#else
    // From the saved stack pointer up: the control words, r15 r14 r13 r12 rbx
    // rbp, the return address. After the return the stack is 16-aligned, as
    // a call wants it.
    auto* frame = reinterpret_cast<std::uint64_t*>(top - 16 - 8 * 8);
    std::memset(frame, 0, 8 * 8);
    // [0] the control words, [1] r15, [2] r14, [3] r13, [4] r12, [5] rbx,
    // [6] rbp, [7] where the swap returns.
    frame[0] = 0x037F00001F80ull;                               // MXCSR's default, then x87's
    frame[3] = reinterpret_cast<std::uint64_t>(parameter);      // r13
    frame[4] = reinterpret_cast<std::uint64_t>(entry);          // r12
    frame[7] = reinterpret_cast<std::uint64_t>(&ps2_fiber_start);
    f->sp = frame;
#endif
    return f;
}

void FiberSwitch(Fiber to) {
    FiberImpl* from = t_current;
    FiberImpl* next = static_cast<FiberImpl*>(to);
    if (from == next) return;
    t_current = next;
    ps2_fiber_swap(&from->sp, next->sp);
}

void FiberDelete(Fiber fiber) {
    auto* f = static_cast<FiberImpl*>(fiber);
    if (!f) return;
    if (f->memory) munmap(f->memory, f->mapped);
    delete f;
}

#endif  // !_WIN32

/* ======================================================================== */
/* Time                                                                      */
/* ======================================================================== */

void SleepUntil(std::chrono::steady_clock::time_point when) {
    const auto left = when - std::chrono::steady_clock::now();
    if (left <= std::chrono::steady_clock::duration::zero()) return;
#ifdef _WIN32
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                                       TIMER_ALL_ACCESS);
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -std::max<long long>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() / 100);
        if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_until(when);
}

void RaiseTimerResolution() {
#ifdef _WIN32
    timeBeginPeriod(1);             // sleeps to the millisecond, not Windows' 15.6 ms
    // ...even with the window hidden behind others: Windows 11 drops a
    // process's timer resolution then (a 200 us sleep took 15 ms, a blank
    // seen 14 ms late), unless told not to; and the process not throttled.
    PROCESS_POWER_THROTTLING_STATE process{};
    process.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    process.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    process.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &process, sizeof process);
#endif
}

void FastThread(bool critical) {
#ifdef _WIN32
    // Not throttled (efficiency cores, lowered clocks) when the window is in
    // the background.
    THREAD_POWER_THROTTLING_STATE state{};
    state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &state, sizeof state);
    SetThreadPriority(GetCurrentThread(), critical ? THREAD_PRIORITY_TIME_CRITICAL : THREAD_PRIORITY_ABOVE_NORMAL);
#elif defined(__APPLE__)
    // The performance cores, scheduled first (an iPad's efficiency cores
    // would run the EE at a third of the speed).
    (void)critical;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
    (void)critical;
#endif
}

#if !defined(_WIN32) && !defined(__linux__) && !defined(__ANDROID__)
namespace {
std::mutex& WaitLock() {
    static std::mutex lock;
    return lock;
}
std::condition_variable& WaitCondition() {
    static std::condition_variable condition;
    return condition;
}
}  // namespace
#endif

void WaitValue(std::atomic<std::uint32_t>* value, std::uint32_t seen, unsigned timeout_ms) {
#ifdef _WIN32
    WaitOnAddress(value, &seen, sizeof seen, timeout_ms);
#elif defined(__linux__) || defined(__ANDROID__)
    timespec timeout{time_t(timeout_ms / 1000), long(timeout_ms % 1000) * 1000000L};
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(value), FUTEX_WAIT_PRIVATE, seen, &timeout, nullptr, 0);
#else
    std::unique_lock<std::mutex> hold(WaitLock());
    WaitCondition().wait_for(hold, std::chrono::milliseconds(timeout_ms),
                             [&] { return value->load(std::memory_order_acquire) != seen; });
#endif
}

void WakeValue(std::atomic<std::uint32_t>* value) {
#ifdef _WIN32
    WakeByAddressAll(value);
#elif defined(__linux__) || defined(__ANDROID__)
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(value), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
#else
    (void)value;
    { std::lock_guard<std::mutex> hold(WaitLock()); }   // a waiter between its check and its wait sees it
    WaitCondition().notify_all();
#endif
}

/* ======================================================================== */
/* The machine's state                                                       */
/* ======================================================================== */

void MemoryUse(std::uint64_t* footprint_mb, std::uint64_t* available_mb) {
    *footprint_mb = *available_mb = 0;
#ifdef __APPLE__
    // The footprint is what iOS counts against the app (its own pages, the
    // GPU's for its images and buffers): the number the Xcode gauge shows.
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        *footprint_mb = info.phys_footprint >> 20;
#if TARGET_OS_IPHONE
    *available_mb = std::uint64_t(os_proc_available_memory()) >> 20;    // before iOS ends the app for memory
#endif
#elif defined(__linux__) || defined(__ANDROID__)
    if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
        unsigned long long size = 0, resident = 0;
        if (std::fscanf(f, "%llu %llu", &size, &resident) == 2)
            *footprint_mb = resident * std::uint64_t(sysconf(_SC_PAGESIZE)) >> 20;
        std::fclose(f);
    }
#endif
}

const char* ThermalState() {
#ifdef __APPLE__
    // [[NSProcessInfo processInfo] thermalState], through the Objective-C
    // runtime (this file is C++).
    Class process_info = objc_getClass("NSProcessInfo");
    if (!process_info) return "";
    using IdCall = id (*)(id, SEL);
    using LongCall = long (*)(id, SEL);
    id info = reinterpret_cast<IdCall>(objc_msgSend)(reinterpret_cast<id>(process_info), sel_registerName("processInfo"));
    if (!info) return "";
    const long state = reinterpret_cast<LongCall>(objc_msgSend)(info, sel_registerName("thermalState"));
    static const char* const kNames[] = {"nominal", "fair", "serious", "critical"};
    return state >= 0 && state < 4 ? kNames[state] : "?";
#else
    return "";
#endif
}

/* ======================================================================== */
/* Files and the calendar                                                    */
/* ======================================================================== */

bool LocalTime(std::time_t t, std::tm* out) {
#ifdef _WIN32
    return localtime_s(out, &t) == 0;
#else
    return localtime_r(&t, out) != nullptr;
#endif
}

bool UtcTime(std::time_t t, std::tm* out) {
#ifdef _WIN32
    return gmtime_s(out, &t) == 0;
#else
    return gmtime_r(&t, out) != nullptr;
#endif
}

int Seek64(std::FILE* f, std::int64_t offset, int origin) {
#ifdef _WIN32
    return _fseeki64(f, offset, origin);
#else
    return fseeko(f, off_t(offset), origin);
#endif
}

std::int64_t Tell64(std::FILE* f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return std::int64_t(ftello(f));
#endif
}

std::FILE* OpenFile(const std::filesystem::path& path, const char* mode) {
#ifdef _WIN32
    std::wstring wide;
    for (const char* m = mode; *m; ++m) wide += wchar_t(*m);
    return _wfopen(path.wstring().c_str(), wide.c_str());
#else
    return std::fopen(path.string().c_str(), mode);
#endif
}

int CompareNoCase(const char* a, const char* b) {
#ifdef _WIN32
    return _stricmp(a, b);
#else
    return strcasecmp(a, b);
#endif
}

std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    return std::filesystem::path(exe).parent_path();
#elif defined(__APPLE__)
    char buffer[4096];
    std::uint32_t size = sizeof buffer;
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        std::error_code ec;
        const auto real = std::filesystem::canonical(buffer, ec);
        return (ec ? std::filesystem::path(buffer) : real).parent_path();
    }
    return std::filesystem::current_path();
#else
    std::error_code ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::current_path() : exe.parent_path();
#endif
}

std::filesystem::path UserDirectory() {
#if defined(__APPLE__) && defined(PS2HOST_IOS)
    // The app's own Documents folder: the Files app shows it ("On My iPad"),
    // so the player puts the disc image there and finds the saves there too.
    if (const char* home = std::getenv("HOME")) {
        std::filesystem::path documents = std::filesystem::path(home) / "Documents";
        std::error_code ec;
        std::filesystem::create_directories(documents, ec);
        return documents;
    }
    return std::filesystem::current_path();
#elif defined(__ANDROID__) && defined(PS2HOST_VULKAN)
    if (const char* external = SDL_GetAndroidExternalStoragePath()) return external;
    if (const char* internal = SDL_GetAndroidInternalStoragePath()) return internal;
    return std::filesystem::current_path();
#else
    return ExecutableDirectory();
#endif
}

void ShowMessage(const std::string& title, const std::string& text, bool error) {
#ifdef _WIN32
    MessageBoxA(nullptr, text.c_str(), title.c_str(), MB_OK | (error ? MB_ICONERROR : MB_ICONWARNING));
#elif defined(PS2HOST_VULKAN)
    if (!SDL_ShowSimpleMessageBox(error ? SDL_MESSAGEBOX_ERROR : SDL_MESSAGEBOX_WARNING, title.c_str(), text.c_str(),
                                  nullptr))
        std::fprintf(stderr, "%s: %s\n", title.c_str(), text.c_str());
#else
    (void)error;
    std::fprintf(stderr, "%s: %s\n", title.c_str(), text.c_str());
#endif
}

}  // namespace platform
}  // namespace ps2host
