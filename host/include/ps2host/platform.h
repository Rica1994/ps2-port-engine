/*
 * What the host asks of the operating system, in one place: Windows' own calls
 * there (the PC port as it always was), POSIX and SDL3 elsewhere (Linux, macOS,
 * iOS/iPadOS, Android).
 *
 *   - fibers: each guest thread's own native stack, switched to by hand;
 *   - precise sleeps, and a wait on a counter another thread bumps;
 *   - the calendar, 64-bit file offsets, files named by std::filesystem::path;
 *   - where the port keeps its settings and saves, and messages to the player.
 */

#ifndef PS2HOST_PLATFORM_H
#define PS2HOST_PLATFORM_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>

namespace ps2host {
namespace platform {

/* ---- Fibers ---------------------------------------------------------------
   A fiber runs `entry(parameter)` on a stack of its own; FiberSwitch saves the
   running one and resumes another. The thread that switches first becomes a
   fiber itself (FiberFromThread). An entry never returns: it switches away
   for the last time. Exceptions are thrown and caught inside one fiber. */
using Fiber = void*;
using FiberEntry = void (*)(void* parameter);
Fiber FiberFromThread();
Fiber FiberCreate(std::size_t stack_bytes, FiberEntry entry, void* parameter);
void FiberSwitch(Fiber to);
void FiberDelete(Fiber fiber);

/* ---- Time -------------------------------------------------------------- */
/* Asleep until `when`, to a tenth of a millisecond or so where the system
   can (a high-resolution timer), sleep_until elsewhere. */
void SleepUntil(std::chrono::steady_clock::time_point when);
/* Sleeps to the millisecond (Windows' timer resolution raised; elsewhere
   they already are), and the process never throttled for being in the
   background behind other windows. */
void RaiseTimerResolution();
/* The calling thread before others: `critical` for the ones asleep almost
   always that must wake on time (the vertical blank's timer, the
   presenter); otherwise above normal. */
void FastThread(bool critical);

/* Waits while `*value == seen`, for up to `timeout_ms`; WakeValue wakes every
   waiter on it (WaitOnAddress / WakeByAddressAll on Windows). */
void WaitValue(std::atomic<std::uint32_t>* value, std::uint32_t seen, unsigned timeout_ms);
void WakeValue(std::atomic<std::uint32_t>* value);

/* ---- Files and the calendar ------------------------------------------ */
bool LocalTime(std::time_t t, std::tm* out);
bool UtcTime(std::time_t t, std::tm* out);
int Seek64(std::FILE* f, std::int64_t offset, int origin);
std::int64_t Tell64(std::FILE* f);
std::FILE* OpenFile(const std::filesystem::path& path, const char* mode);
int CompareNoCase(const char* a, const char* b);

/* The running executable's directory. */
std::filesystem::path ExecutableDirectory();
/* Where the player's own files go: the settings, the memory card, the log.
   Beside the executable on a PC; the app's Documents folder on iOS/iPadOS
   (shown in the Files app), its internal storage on Android. */
std::filesystem::path UserDirectory();

/* A message to the player, in a box (there is no terminal to read). */
void ShowMessage(const std::string& title, const std::string& text, bool error);

/* ---- The machine's state, for a tablet's log ------------------------- */
/* The process's memory footprint, and - on iOS - what the system still
   allows it before ending it for memory; in MB, 0 where unknown. */
void MemoryUse(std::uint64_t* footprint_mb, std::uint64_t* available_mb);
/* The C/C++ heap's bytes in use (malloc's zones; 0 where it cannot say). */
std::uint64_t HeapInUse();
/* The device's thermal state as the system tells it (Apple's: nominal, fair,
   serious, critical - the last two slow the cores and the GPU); "" where
   there is none to tell. */
const char* ThermalState();

/* Whether this is a phone or tablet build: no files beside the executable,
   one window filling the screen, touch. */
constexpr bool kMobile =
#if defined(__ANDROID__) || (defined(__APPLE__) && defined(PS2HOST_IOS))
    true;
#else
    false;
#endif

}  // namespace platform
}  // namespace ps2host

#endif
