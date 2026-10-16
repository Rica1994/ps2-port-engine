/*
 * A tablet's reports, sent by themselves: the run's log and the VU programs
 * it recorded, over the local network to a receiver on the developer's PC
 * (engine/tools/ipad_reports.py), so that nobody carries files out of the
 * Files app. Every fifteen seconds whatever changed, and once more at the
 * end of the run.
 *
 * Where to: PS2HOST_REPORT (an Xcode scheme's environment) or the build's
 * own address (CMake's PS2_REPORT_URL, which engine/tools/ipad_update.sh
 * sets): "http://192.168.1.20:8765". Neither: nothing is sent. A receiver
 * that is not there costs a failed connection every fifteen seconds, on a
 * thread of its own.
 *
 * Plain HTTP over a socket (a POST of each file's bytes) - the receiver is on
 * the same network, and nothing here needs more. iOS asks the player once
 * whether the app may use the local network (Info.plist's
 * NSLocalNetworkUsageDescription); until then the sends fail and are tried
 * again.
 */
#include "ps2host/run.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__linux__) || defined(__ANDROID__)
#define PS2HOST_REPORTS 1
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#if defined(PS2HOST_REPORTS) && __has_include(<SDL3/SDL_events.h>)
#include <SDL3/SDL_events.h>
#define PS2HOST_REPORTS_SDL 1
#endif

#ifndef PS2HOST_REPORT_DEFAULT
#define PS2HOST_REPORT_DEFAULT ""
#endif

namespace ps2host {

namespace {

namespace fs = std::filesystem;

struct Sent {
    std::uintmax_t size = 0;
    fs::file_time_type when{};
};

struct Reports {
    std::string host, port;
    std::vector<std::pair<fs::path, std::string>> files;    // the file, its name at the receiver
    std::map<std::string, Sent> sent;
    std::mutex lock;                                        // one send at a time
    std::mutex wake_lock;
    std::condition_variable wake;
    bool stopping = false;
    bool send_now = false;
    std::thread thread;
    bool told = false;                                      // the first failure said in the log
};

Reports* g_reports = nullptr;

#ifdef PS2HOST_REPORTS
/* A connection to host:port within `timeout_ms`, or -1. */
int Connect(const std::string& host, const std::string& port, int timeout_ms) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &found) != 0 || !found) return -1;
    int fd = -1;
    for (addrinfo* a = found; a && fd < 0; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
#ifdef SO_NOSIGPIPE
        const int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int r = connect(fd, a->ai_addr, a->ai_addrlen);
        if (r != 0 && errno == EINPROGRESS) {
            pollfd p{fd, POLLOUT, 0};
            r = poll(&p, 1, timeout_ms) == 1 ? 0 : -1;
            int error = 0;
            socklen_t length = sizeof error;
            if (r == 0 && (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) != 0 || error)) r = -1;
        }
        if (r != 0) {
            close(fd);
            fd = -1;
            continue;
        }
        fcntl(fd, F_SETFL, flags);
        timeval t{timeout_ms / 1000 * 5, 0};                // sends of a large log take longer than a connect
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof t);
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof t);
    }
    freeaddrinfo(found);
    return fd;
}

bool SendAll(int fd, const char* data, std::size_t size) {
    while (size) {
#ifdef MSG_NOSIGNAL
        const ssize_t n = send(fd, data, size, MSG_NOSIGNAL);
#else
        const ssize_t n = send(fd, data, size, 0);
#endif
        if (n <= 0) return false;
        data += n;
        size -= std::size_t(n);
    }
    return true;
}

/* POST /upload/<name> with the bytes; true on the receiver's 200. */
bool Post(Reports& r, const std::string& name, const std::string& body, std::string* why) {
    const int fd = Connect(r.host, r.port, 2000);
    if (fd < 0) {
        *why = "no connection to " + r.host + ":" + r.port;
        return false;
    }
    const std::string head = "POST /upload/" + name + " HTTP/1.1\r\nHost: " + r.host + ":" + r.port +
                             "\r\nContent-Type: application/octet-stream\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    bool ok = SendAll(fd, head.data(), head.size()) && SendAll(fd, body.data(), body.size());
    if (ok) {
        // The whole reply read (closing with some of it unread resets the
        // connection under the receiver's feet), its status line checked.
        std::string reply;
        char chunk[512];
        for (ssize_t n; (n = recv(fd, chunk, sizeof chunk, 0)) > 0 && reply.size() < 4096;)
            reply.append(chunk, std::size_t(n));
        ok = reply.size() > 12 && reply.compare(0, 7, "HTTP/1.") == 0 && reply.compare(9, 3, "200") == 0;
        if (!ok) *why = reply.empty() ? "no reply" : reply.substr(0, std::min<std::size_t>(reply.size(), 12));
    } else {
        *why = "the send failed";
    }
    close(fd);
    return ok;
}
#endif

/* Every file that changed since it was last sent, sent. */
void SendChanged(Reports& r) {
#ifdef PS2HOST_REPORTS
    std::lock_guard<std::mutex> hold(r.lock);
    for (const auto& [path, name] : r.files) {
        std::error_code error;
        const std::uintmax_t size = fs::file_size(path, error);
        if (error) continue;
        const fs::file_time_type when = fs::last_write_time(path, error);
        if (error) continue;
        Sent& last = r.sent[name];
        if (size == last.size && when == last.when) continue;
        std::ifstream in(path, std::ios::binary);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string why;
        if (Post(r, name, body, &why)) {
            last.size = size;
            last.when = when;
            if (r.told) std::fprintf(stderr, "[report] sent again (%s)\n", name.c_str());
            r.told = false;
        } else if (!r.told) {
            std::fprintf(stderr, "[report] %s not sent: %s (tried again every 15 s)\n", name.c_str(), why.c_str());
            r.told = true;
        }
    }
#else
    (void)r;
#endif
}

#ifdef PS2HOST_REPORTS_SDL
/* The app leaving the front (iOS gives it a few seconds before it is
   suspended): the reports go now, from their thread. */
bool SDLCALL OnAppEvent(void*, SDL_Event* e) {
    if (e->type == SDL_EVENT_WILL_ENTER_BACKGROUND || e->type == SDL_EVENT_TERMINATING) {
        if (Reports* r = g_reports) {
            std::lock_guard<std::mutex> hold(r->wake_lock);
            r->send_now = true;
            r->wake.notify_all();
        }
    }
    return true;
}
#endif

/* "http://host:port[/]" -> host, port. */
bool ParseUrl(std::string url, std::string* host, std::string* port) {
    const std::string scheme = "http://";
    if (url.compare(0, scheme.size(), scheme) == 0) url = url.substr(scheme.size());
    while (!url.empty() && url.back() == '/') url.pop_back();
    const std::size_t colon = url.rfind(':');
    *host = colon == std::string::npos ? url : url.substr(0, colon);
    *port = colon == std::string::npos ? "8765" : url.substr(colon + 1);
    return !host->empty() && !port->empty();
}

}  // namespace

void StartReports(const fs::path& log, const fs::path& folder, const std::string& run_name) {
#ifdef PS2HOST_REPORTS
    const char* env = std::getenv("PS2HOST_REPORT");
    const std::string url = env && *env ? env : PS2HOST_REPORT_DEFAULT;
    if (url.empty() || url == "0" || g_reports) return;
    auto* r = new Reports;
    if (!ParseUrl(url, &r->host, &r->port)) {
        std::fprintf(stderr, "[report] not an address: %s\n", url.c_str());
        delete r;
        return;
    }
    r->files.push_back({log, run_name + ".log"});
    r->files.push_back({folder / "vu_programs_new.txt", "vu_programs_new.txt"});
    g_reports = r;
    std::fprintf(stderr, "[report] the log and the recorded VU programs go to %s:%s\n", r->host.c_str(),
                 r->port.c_str());
    r->thread = std::thread([r] {
        std::unique_lock<std::mutex> hold(r->wake_lock);
        while (!r->stopping) {
            r->wake.wait_for(hold, std::chrono::seconds(15), [r] { return r->send_now || r->stopping; });
            if (r->stopping) break;
            r->send_now = false;
            hold.unlock();
            SendChanged(*r);
            hold.lock();
        }
    });
    r->thread.detach();
    std::atexit([] { SendReportsNow(); });
#ifdef PS2HOST_REPORTS_SDL
    SDL_AddEventWatch(OnAppEvent, nullptr);
#endif
#else
    (void)log;
    (void)folder;
    (void)run_name;
#endif
}

void SendReportsNow() {
    if (!g_reports) return;
    std::fflush(stderr);
    SendChanged(*g_reports);
}

}  // namespace ps2host
