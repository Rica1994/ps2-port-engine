/*
 * Every game port's entry point. The game supplies its translated functions (the
 * table engine/tools/port.py generates) and its overrides; the rest is shared.
 *
 * On Windows the port is a windowed program: it asks for the disc image with a
 * file dialog the first time. On iOS/iPadOS (and Android) it is an app: the
 * player copies the disc image into the app's folder in the Files app, and the
 * port finds it there. Elsewhere (Linux, macOS) it takes the disc image as an
 * argument, or the one named in pc_settings.ini, or one beside it.
 */

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(PS2HOST_VULKAN) && !defined(_WIN32)
// SDL's own entry point where the system wants one (iOS's UIApplicationMain,
// Android's activity): main() below becomes SDL_main.
#include <SDL3/SDL_main.h>
#endif

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/platform.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

extern const ps2host::FunctionEntry kGuestFunctions[];
extern const std::size_t kGuestFunctionCount;
extern const ps2host::OverlayEntry kGuestOverlays[];
extern const std::size_t kGuestOverlayCount;

/* The game's host implementations of guest functions (its port's overrides.cpp). */
void RegisterGameOverrides();

namespace {

namespace platform = ps2host::platform;

/* The port is built for AVX2 on Windows (engine/host/CMakeLists.txt; the VU's
   lanes and the rest take its three-operand forms): a processor without it is
   told so rather than stopped by an illegal instruction somewhere. */
bool HasAvx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)) && defined(__AVX2__)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool osxsave = (info[2] >> 27) & 1, avx = (info[2] >> 28) & 1;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false;        // the OS saves the YMM registers
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    return true;
#endif
}

std::string g_log;     // the log a run started without arguments writes

#ifdef PS2HOST_GAME_EXECUTABLE
/* Whether `path` is an image of the game's disc: the executable it boots is
   the one the port was translated from (game.json's). */
bool IsGameDisc(const std::string& path) {
    if (path.empty()) return false;
    ps2host::Disc disc;
    if (!disc.Open(path)) return false;
    std::string boot = disc.BootPath();
    const std::size_t semicolon = boot.find(';');
    if (semicolon != std::string::npos) boot.resize(semicolon);
    while (!boot.empty() && (boot[0] == '\\' || boot[0] == '/')) boot.erase(0, 1);
    return platform::CompareNoCase(boot.c_str(), PS2HOST_GAME_EXECUTABLE) == 0;
}

bool IsDiscImageName(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".iso" || ext == ".cue" || ext == ".bin";
}

/* The first image of the game's disc in `folder` (and the folders in it, one
   level down): where the player put it with the Files app. */
std::string FindDisc(const std::filesystem::path& folder) {
    std::error_code ec;
    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
        if (entry.is_regular_file(ec) && IsDiscImageName(entry.path())) candidates.push_back(entry.path());
        else if (entry.is_directory(ec))
            for (const auto& inner : std::filesystem::directory_iterator(entry.path(), ec))
                if (inner.is_regular_file(ec) && IsDiscImageName(inner.path())) candidates.push_back(inner.path());
    }
    for (const auto& c : candidates)
        if (IsGameDisc(c.string())) return c.string();
    return {};
}

#ifdef _WIN32
/* The player's disc image, from a file dialog; "" if they cancel. */
std::string PickDisc() {
    char file[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof dialog;
    dialog.lpstrFilter = "Disc images (*.iso; *.cue; *.bin)\0*.iso;*.cue;*.bin\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrTitle = "Choose your " PS2HOST_GAME_NAME " disc image";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&dialog) ? std::string(file) : std::string();
}
#endif

/* What the run says, to a log beside the settings: there is no terminal to
   read. Opened once and given to both streams: their lines in order. */
void LogToFile(const std::filesystem::path& here, const std::string& name) {
    g_log = name;
    const std::string log = (here / g_log).string();
#ifdef _WIN32
    std::FILE* stream = nullptr;
    freopen_s(&stream, "NUL", "w", stdout);    // streams with descriptors, in a windowed program
    freopen_s(&stream, "NUL", "w", stderr);
    const int file = _open(log.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_TEXT, _S_IREAD | _S_IWRITE);
    if (file >= 0) {
        _dup2(file, _fileno(stdout));
        _dup2(file, _fileno(stderr));
        _close(file);
    }
#else
    const int file = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (file >= 0) {
        dup2(file, fileno(stdout));
        dup2(file, fileno(stderr));
        close(file);
    }
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a run ended by force loses nothing
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}
#endif

/* Started without arguments - from Explorer, a shortcut, the home screen - the
   port plays its game from the player's own disc image: the one chosen before
   (kept in the PC settings), or asked for (Windows) or found (a tablet, in the
   app's folder) and checked to be the game's disc; the memory card a
   directory beside the settings. In a window, with sound, on the GPU, with no
   time limit (the default is for test runs). An empty list: the player gave
   none. */
std::vector<std::string> PlayArguments(const char* self) {
#ifdef PS2HOST_GAME_EXECUTABLE
    const std::filesystem::path here = platform::UserDirectory();
    ps2host::LoadSettings((here / "pc_settings.ini").string());     // where --memcard puts them
    std::string disc = ps2host::Setting("disc");
#ifdef _WIN32
    // None chosen yet, or gone: an image of the game's disc beside the
    // executable is taken without asking.
    if (!IsGameDisc(disc)) {
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(here, error)) {
            std::string ext = entry.path().extension().string();
            for (char& ch : ext) ch = char(tolower(static_cast<unsigned char>(ch)));
            if ((ext == ".iso" || ext == ".bin" || ext == ".cue") && IsGameDisc(entry.path().string())) {
                disc = entry.path().string();
                break;
            }
        }
    }
    while (!IsGameDisc(disc)) {
        std::error_code ec;
        if (!disc.empty() && std::filesystem::exists(disc, ec))
            platform::ShowMessage(PS2HOST_GAME_NAME, "This is not an image of the " PS2HOST_GAME_NAME " disc "
                                                     "(" PS2HOST_GAME_EXECUTABLE "):\n\n" + disc, false);
        disc = PickDisc();
        if (disc.empty()) return {};
    }
#else
    if (!IsGameDisc(disc)) disc = FindDisc(here);
    if (disc.empty()) {
        platform::ShowMessage(PS2HOST_GAME_NAME,
                              platform::kMobile
                                  ? "Copy your own disc image of " PS2HOST_GAME_NAME " (" PS2HOST_GAME_EXECUTABLE
                                    ", an .iso) into this app's folder with the Files app (On My iPad > "
                                    "the game's folder), then start the game again."
                                  : "Put your own disc image of " PS2HOST_GAME_NAME " (" PS2HOST_GAME_EXECUTABLE
                                    ", an .iso) beside the game, or name it: --disc <image>.",
                              false);
        return {};
    }
#endif
    if (disc != ps2host::Setting("disc")) ps2host::SetSetting("disc", disc);
    std::string stem = std::filesystem::path(self ? self : "port").stem().string();
    if (stem.empty()) stem = "port";
    // PS2HOST_SCRIPT="<arguments>" (set in an Xcode scheme's environment, say):
    // a scripted run - a port/runs/*.args, --virtual-time and its presses - on
    // the player's own disc and settings, with no memory card (the runs were
    // recorded with none): a tablet's benchmark, its numbers in a log of its
    // own (burnout3-20261016-093000.log), so that runs one after another can
    // be compared. A tablet records the VU programs it starts in either kind
    // of run, for the next build to compile ahead.
    const char* script = std::getenv("PS2HOST_SCRIPT");
    char when[32] = "run";
    {
        std::tm t{};
        if (platform::LocalTime(std::time(nullptr), &t)) std::strftime(when, sizeof when, "%Y%m%d-%H%M%S", &t);
    }
    std::string log = stem + ".log";
    if (script && *script) log = stem + "-" + when + ".log";
    LogToFile(here, log);
    // The log and the recorded programs to the developer's PC as the run goes
    // (report.cpp), each run's log under its own name there.
    ps2host::StartReports(here / log, here, stem + "-" + when + (script && *script ? "-scripted" : ""));
    std::fprintf(stderr, "[port] disc %s\n", disc.c_str());
    if (script && *script) {
        std::vector<std::string> args = {self ? self : "port", "--disc", disc, "--window", "--gpu", "--audio",
                                         "--timeout", "315360000", "--frame-times", "300"};
        if (platform::kMobile) {
            args.push_back("--vu-record");
            args.push_back((here / "vu_programs_new.txt").string());
        }
        std::string word;
        for (const char* c = script;; ++c) {
            if (*c && !std::isspace(static_cast<unsigned char>(*c))) {
                word += *c;
                continue;
            }
            // A script's own settings for the run, as the environment's
            // (one line in an Xcode scheme does it all): --skip-cycle=...
            // (PS2HOST_SKIP_CYCLE: what each drawing function costs),
            // --frameskip=0/1 (PS2HOST_FRAMESKIP, a port's frame skipping).
            const auto take = [&](const char* option, const char* variable) {
                const std::string prefix = std::string(option) + "=";
                if (word.compare(0, prefix.size(), prefix) != 0) return false;
                platform::SetEnvironment(variable, word.substr(prefix.size()).c_str());
                return true;
            };
            if (!word.empty() && !take("--skip-cycle", "PS2HOST_SKIP_CYCLE") &&
                !take("--skip-seconds", "PS2HOST_SKIP_SECONDS") &&
                !take("--skip-held", "PS2HOST_SKIP_HELD") && !take("--frameskip", "PS2HOST_FRAMESKIP"))
                args.push_back(word);
            word.clear();
            if (!*c) break;
        }
        std::fprintf(stderr, "[port] scripted run: %s\n", script);
        return args;
    }
    return {self ? self : "port", "--disc", disc, "--memcard", (here / "memcard").string(), "--window", "--gpu",
            "--audio", "--timeout", "315360000",     // ten years
            "--frame-times", platform::kMobile ? "300" : "0",   // a tablet's log: pictures a second
            // A tablet records the VU programs it plays that are not compiled
            // ahead: sent back, they are compiled into the next build.
            "--vu-record", platform::kMobile ? (here / "vu_programs_new.txt").string() : std::string()};
#else
    (void)self;
    return {self ? self : "port"};
#endif
}

#ifdef _WIN32
/* A port built as a windowed program has no console of its own. Started from
   a terminal its output goes to that one; redirected (a file, a pipe: the
   tools' runs), where it was sent. */
void AttachTerminal() {
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE) return;
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    std::FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    AttachTerminal();
#endif
    std::vector<std::string> play;
    std::vector<char*> play_argv;
    if (argc <= 1) {
        play = PlayArguments(argc > 0 ? argv[0] : nullptr);
        if (play.size() <= 1) return 0;
        for (std::string& arg : play) play_argv.push_back(arg.data());
        play_argv.push_back(nullptr);
        argc = int(play.size());
        argv = play_argv.data();
    }
    if (!HasAvx2()) {
        std::fprintf(stderr, "This build needs a processor with AVX2 (Intel Haswell, AMD Excavator or later).\n");
        return 70;
    }
#ifdef PS2HOST_GAME_TITLE
    ps2host::SetGameTitle(PS2HOST_GAME_TITLE);      // the window's (game.json's "title")
#endif
#ifdef PS2HOST_GAME_VERSION
    std::fprintf(stderr, "[port] %s, version %s\n", PS2HOST_GAME_TITLE, PS2HOST_GAME_VERSION);
#endif
    ps2host::RegisterFunctions(kGuestFunctions, kGuestFunctionCount);
    ps2host::RegisterOverlays(kGuestOverlays, kGuestOverlayCount);
    RegisterGameOverrides();
    const int code = ps2host::Main(argc, argv);
    if (!play.empty() && code != 0) {         // closing the window ends a run with 0
        std::fflush(nullptr);
        platform::ShowMessage(ps2host::GameTitle().empty() ? "PS2 port" : ps2host::GameTitle(),
                              "The game stopped because of an error. What happened is in " + g_log +
                                  ", beside the game.",
                              true);
    }
    // What must be finished is (the report written, the settings and the
    // memory card saved as they changed, the WAV's header here); the rest -
    // the window, the GPU device, the threads - goes with the process.
    // Taken down one by one at exit, the Vulkan device's could hang
    // (vkDestroyDevice with the driver's threads and the renderer's images
    // still about), leaving a window that never closed.
    ps2host::CloseAudio();
    ps2host::ReleaseController();
    std::fflush(nullptr);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), UINT(code));
#else
    _exit(code);
#endif
    return code;
}
