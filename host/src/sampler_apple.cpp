/*
 * A tablet's own profiler (iOS): the threads that do the work - the EE's, the
 * GS thread, VIF1's worker - sampled a thousand times a second, and every ten
 * seconds the functions they were found in written to the log, by name. On an
 * iPad there is no terminal and Instruments may not name the port's functions;
 * the log is what comes back.
 *
 * Each sample suspends the thread, reads its registers and walks its frame
 * pointers (arm64 keeps them), then lets it go: nothing is allocated while it
 * is suspended (it may hold the allocator's lock). Names come from the image's
 * own symbol table (__LINKEDIT, mapped with the image: a build run from Xcode
 * is not stripped), else dladdr (the system's exported names).
 */

#include "ps2host/profile.h"

#if defined(__APPLE__) && defined(PS2HOST_IOS)

#include <cxxabi.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach/mach.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ps2host {

namespace {

constexpr int kDepth = 12;
constexpr std::uint64_t kAddressMask = 0x0000007FFFFFFFFFull;     // pointer authentication's bits off

struct Target {
    std::string name;
    thread_act_t port;
    std::uintptr_t stack_low, stack_high;
};

struct Sample {
    int target;
    std::uint64_t pc[kDepth];
};

std::mutex g_lock;
std::vector<Target> g_targets;
std::atomic<bool> g_started{false};

// ---------------------------------------------------------------- names

struct Image {
    std::vector<std::pair<std::uint64_t, const char*>> symbols;     // sorted by address
    std::string short_name;
};

std::unordered_map<std::uintptr_t, Image> g_images;   // by load address

const Image& ImageAt(const void* base, const char* path) {
    auto found = g_images.find(std::uintptr_t(base));
    if (found != g_images.end()) return found->second;
    Image& image = g_images[std::uintptr_t(base)];
    if (path) {
        const char* slash = std::strrchr(path, '/');
        image.short_name = slash ? slash + 1 : path;
    }
    const auto* header = static_cast<const mach_header_64*>(base);
    if (header->magic != MH_MAGIC_64) return image;
    const segment_command_64 *text = nullptr, *linkedit = nullptr;
    const symtab_command* symtab = nullptr;
    const auto* command = reinterpret_cast<const load_command*>(header + 1);
    for (std::uint32_t i = 0; i < header->ncmds; ++i) {
        if (command->cmd == LC_SEGMENT_64) {
            const auto* segment = reinterpret_cast<const segment_command_64*>(command);
            if (std::strcmp(segment->segname, SEG_TEXT) == 0) text = segment;
            if (std::strcmp(segment->segname, SEG_LINKEDIT) == 0) linkedit = segment;
        } else if (command->cmd == LC_SYMTAB) {
            symtab = reinterpret_cast<const symtab_command*>(command);
        }
        command = reinterpret_cast<const load_command*>(reinterpret_cast<const char*>(command) + command->cmdsize);
    }
    if (!text || !linkedit || !symtab) return image;
    const std::uintptr_t slide = std::uintptr_t(base) - text->vmaddr;
    const std::uintptr_t linkedit_base = slide + linkedit->vmaddr - linkedit->fileoff;
    const auto* entries = reinterpret_cast<const nlist_64*>(linkedit_base + symtab->symoff);
    const char* strings = reinterpret_cast<const char*>(linkedit_base + symtab->stroff);
    for (std::uint32_t i = 0; i < symtab->nsyms; ++i) {
        const nlist_64& n = entries[i];
        if ((n.n_type & N_STAB) || (n.n_type & N_TYPE) != N_SECT || !n.n_value) continue;
        image.symbols.push_back({n.n_value + slide, strings + n.n_un.n_strx});
    }
    std::sort(image.symbols.begin(), image.symbols.end());
    return image;
}

std::string Demangled(const char* name) {
    if (!name) return "?";
    const char* mangled = name[0] == '_' ? name + 1 : name;
    int status = 0;
    char* text = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    std::string out = status == 0 && text ? text : mangled;
    std::free(text);
    // Templates' arguments make the lines unreadable: the name alone.
    if (out.size() > 110) out = out.substr(0, 110) + "...";
    return out;
}

std::unordered_map<std::uint64_t, std::string> g_names;

const std::string& NameOf(std::uint64_t address) {
    auto found = g_names.find(address);
    if (found != g_names.end()) return found->second;
    std::string text;
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(address), &info) && info.dli_fbase) {
        const Image& image = ImageAt(info.dli_fbase, info.dli_fname);
        const char* best = nullptr;
        if (!image.symbols.empty()) {
            auto it = std::upper_bound(image.symbols.begin(), image.symbols.end(),
                                       std::make_pair(address, static_cast<const char*>(nullptr)),
                                       [](const auto& a, const auto& b) { return a.first < b.first; });
            if (it != image.symbols.begin()) best = std::prev(it)->second;
        }
        if (!best) best = info.dli_sname;
        text = Demangled(best) + "  [" + image.short_name + "]";
    } else {
        char hex[32];
        std::snprintf(hex, sizeof hex, "0x%llx", static_cast<unsigned long long>(address));
        text = hex;
    }
    return g_names.emplace(address, text).first->second;
}

// ---------------------------------------------------------------- sampling

void Report(const std::vector<Sample>& samples, double seconds) {
    std::lock_guard<std::mutex> hold(g_lock);
    for (int t = 0; t < int(g_targets.size()); ++t) {
        std::map<std::string, std::uint64_t> self, total;
        std::map<std::string, std::map<std::string, std::uint64_t>> callers;
        std::uint64_t count = 0;
        for (const Sample& s : samples) {
            if (s.target != t || !s.pc[0]) continue;
            ++count;
            const std::string& leaf = NameOf(s.pc[0]);
            ++self[leaf];
            if (s.pc[1]) ++callers[leaf][NameOf(s.pc[1] - 4)];
            std::vector<const std::string*> seen;
            for (int d = 0; d < kDepth && s.pc[d]; ++d) {
                const std::string& fn = NameOf(d ? s.pc[d] - 4 : s.pc[d]);
                if (std::find(seen.begin(), seen.end(), &fn) != seen.end()) continue;
                seen.push_back(&fn);
                ++total[fn];
            }
        }
        if (!count) continue;
        auto ranked = [](const std::map<std::string, std::uint64_t>& m) {
            std::vector<std::pair<std::uint64_t, std::string>> v;
            for (const auto& [k, n] : m) v.push_back({n, k});
            std::sort(v.rbegin(), v.rend());
            return v;
        };
        std::fprintf(stderr, "[sample] %s thread: %llu samples in %.0f s (%.0f%% of the time running)\n",
                     g_targets[t].name.c_str(), static_cast<unsigned long long>(count), seconds,
                     100.0 * double(count) / (seconds * 1000.0));
        const auto by_self = ranked(self);
        for (int i = 0; i < 18 && i < int(by_self.size()); ++i) {
            const auto by_caller = ranked(callers[by_self[i].second]);
            std::fprintf(stderr, "[sample]   self %5.1f%%  %s%s%s\n", 100.0 * double(by_self[i].first) / double(count),
                         by_self[i].second.c_str(), by_caller.empty() ? "" : "  <- ",
                         by_caller.empty() ? "" : by_caller[0].second.c_str());
        }
        const auto by_total = ranked(total);
        for (int i = 0; i < 18 && i < int(by_total.size()); ++i)
            std::fprintf(stderr, "[sample]   in   %5.1f%%  %s\n", 100.0 * double(by_total[i].first) / double(count),
                         by_total[i].second.c_str());
    }
    std::fflush(stderr);
}

void SamplerLoop() {
    std::vector<Sample> samples;
    samples.reserve(1 << 16);
    auto last = std::chrono::steady_clock::now();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::vector<Target> targets;
        {
            std::lock_guard<std::mutex> hold(g_lock);
            targets = g_targets;
        }
        for (int t = 0; t < int(targets.size()); ++t) {
            Sample s{t, {}};
            const Target& target = targets[t];
            if (thread_suspend(target.port) != KERN_SUCCESS) continue;
            arm_thread_state64_t state{};
            mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
            bool running = false;
            if (thread_get_state(target.port, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&state), &n) ==
                KERN_SUCCESS) {
                s.pc[0] = arm_thread_state64_get_pc(state) & kAddressMask;
                std::uintptr_t fp = arm_thread_state64_get_fp(state) & kAddressMask;
                s.pc[1] = arm_thread_state64_get_lr(state) & kAddressMask;
                // The caller of a leaf is in LR; above it, the frame chain
                // (its first frame's return address is LR's again: skipped).
                int d = 2;
                bool first = true;
                while (d < kDepth && fp >= target.stack_low && fp + 16 <= target.stack_high && (fp & 7) == 0) {
                    const std::uintptr_t next = *reinterpret_cast<const std::uintptr_t*>(fp) & kAddressMask;
                    const std::uint64_t ret = *reinterpret_cast<const std::uint64_t*>(fp + 8) & kAddressMask;
                    if (!first || ret != s.pc[1]) s.pc[d++] = ret;
                    first = false;
                    if (next <= fp) break;
                    fp = next;
                }
                // A thread asleep in the kernel is not where the time goes.
                thread_basic_info_data_t basic{};
                mach_msg_type_number_t bn = THREAD_BASIC_INFO_COUNT;
                if (thread_info(target.port, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&basic), &bn) ==
                    KERN_SUCCESS)
                    running = basic.run_state == TH_STATE_RUNNING;
            }
            thread_resume(target.port);
            if (running) samples.push_back(s);
        }
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last).count();
        if (seconds >= 10.0) {
            Report(samples, seconds);
            samples.clear();
            last = std::chrono::steady_clock::now();
        }
    }
}

}  // namespace

void SampleThisThread(const char* name) {
    pthread_t self = pthread_self();
    const std::uintptr_t high = std::uintptr_t(pthread_get_stackaddr_np(self));
    const std::uintptr_t low = high - pthread_get_stacksize_np(self);
    {
        std::lock_guard<std::mutex> hold(g_lock);
        g_targets.push_back({name, mach_thread_self(), low, high});
    }
    if (!g_started.exchange(true)) std::thread(SamplerLoop).detach();
}

}  // namespace ps2host

#else

namespace ps2host {
void SampleThisThread(const char*) {}
}  // namespace ps2host

#endif
