#include "ps2host/asset_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace ps2host::assets {

namespace {

// What the store's records were cooked by: a store of another kind is put
// aside, not read (a new encoder, another effort, a changed quality gate).
constexpr std::uint32_t kCookVersion = 3;        // 3: alpha scaled to its top
constexpr char kMagic[8] = {'P', 'S', '2', 'A', 'S', 'S', 'T', '1'};
constexpr std::uint32_t kRecordMagic = 0x54534552;       // "REST"
constexpr std::uint32_t kRejected = 1;                   // a record without data: drawn decoded
// Flags' bits 8-15: Cooked::alpha_top.

struct FileHeader {
    char magic[8];
    std::uint32_t format, stamp;
    std::uint32_t reserved[4];
};
struct RecordHeader {
    std::uint32_t magic, flags;
    std::uint64_t id;
    std::uint32_t format, width, height, bytes, checksum, reserved;
};
static_assert(sizeof(FileHeader) == 32 && sizeof(RecordHeader) == 40, "the store's layout");

std::uint32_t Checksum(const std::uint8_t* p, std::size_t n) {
    std::uint32_t h = 2166136261u;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
    return h;
}

// Alpha stretched so that `top` is 255 (Cooked::alpha_top), and back: each
// value comes back exactly (top is at most 255), 0 and top at the ends of the
// range, where a block-compressed format keeps them.
std::uint32_t Stretch(std::uint32_t a, std::uint32_t top) { return (a * 255 + top / 2) / top; }
std::uint32_t Unstretch(std::uint32_t a, std::uint32_t top) { return (a * top + 127) / 255; }
void UnstretchAlpha(std::vector<std::uint32_t>* rgba, std::uint32_t top) {
    for (std::uint32_t& texel : *rgba) texel = (texel & 0xFFFFFF) | Unstretch(texel >> 24, top) << 24;
}

void Background() {
#if defined(__APPLE__)
    // An iPad's efficiency cores: the game's threads keep the fast ones.
    pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#endif
}

struct Entry {
    std::uint64_t offset = 0;           // the data's, in the store
    std::uint32_t bytes = 0, width = 0, height = 0, checksum = 0;
    codec::Format format = codec::kRgba8;
    bool rejected = false;
    std::uint32_t alpha_top = 0;
};

struct Job {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> rgba;
    std::uint32_t width = 0, height = 0;
};

struct Cache {
    Options options;
    bool open = false;
    std::filesystem::path path;

    std::mutex lock;
    std::condition_variable wake_loader, wake_cooker;
    bool stopping = false;
    std::unordered_map<std::uint64_t, Entry> index;
    std::atomic<std::uint64_t> store_bytes{0};
    // In memory, least recently used last.
    struct Held {
        std::shared_ptr<const Cooked> image;
        std::list<std::uint64_t>::iterator lru;
    };
    std::unordered_map<std::uint64_t, Held> held;
    std::list<std::uint64_t> lru;
    std::uint64_t held_bytes = 0;
    std::deque<std::uint64_t> loads;
    std::unordered_set<std::uint64_t> loading;
    std::deque<Job> cooks;
    std::unordered_set<std::uint64_t> cooking;
    std::uint64_t queued_bytes = 0;     // RGBA waiting to be cooked
    std::thread loader, cooker;
    std::FILE* append = nullptr;        // the cooker's (append_lock)
    std::FILE* read = nullptr;          // the loader's (read_lock)
    std::mutex append_lock, read_lock;

    // The report's counts (since the start; Report takes differences).
    std::atomic<std::uint64_t> cooked{0}, rejected{0}, cooked_in{0}, cooked_out{0}, cook_ns{0}, loaded{0},
        load_failed{0}, found{0}, dropped{0}, prefetched{0};

    /* Prefetch: the order textures were first asked for, in the last run
       (`previous`, and where each is in it) and in this one. A texture asked
       for that the last run asked for at place p brings the next kPrefetch
       after it from the store into memory - a race's textures are read
       before the race asks for them, as a level's bundle is. */
    static constexpr std::size_t kPrefetch = 48;
    std::vector<std::uint64_t> previous;
    std::unordered_map<std::uint64_t, std::uint32_t> previous_at;
    std::vector<std::uint64_t> order;
    std::unordered_set<std::uint64_t> ordered;
    std::size_t order_saved = 0;

    void Prefetch(std::uint64_t id) {                 // (under lock)
        auto at = previous_at.find(id);
        if (at == previous_at.end() || options.synchronous) return;
        const std::size_t end = std::min(previous.size(), std::size_t(at->second) + 1 + kPrefetch);
        for (std::size_t i = at->second + 1; i < end; ++i) {
            const std::uint64_t next = previous[i];
            auto known = index.find(next);
            if (known == index.end() || known->second.rejected || held.count(next) || loading.count(next)) continue;
            loading.insert(next);
            loads.push_back(next);
            ++prefetched;
        }
        wake_loader.notify_one();
    }

    std::filesystem::path OrderPath() const { return path.string() + ".order"; }

    void LoadOrder() {
        std::FILE* f = std::fopen(OrderPath().string().c_str(), "rb");
        if (!f) return;
        std::uint64_t id = 0;
        while (std::fread(&id, sizeof id, 1, f) == 1 && previous.size() < (1u << 20)) {
            if (previous_at.emplace(id, std::uint32_t(previous.size())).second) previous.push_back(id);
        }
        std::fclose(f);
    }

    // This run's order, then what the last run had that this one has not
    // asked for yet (a run that stopped in the menus keeps the race's).
    void SaveOrder() {
        std::vector<std::uint64_t> ids;
        {
            std::lock_guard<std::mutex> hold(lock);
            if (order.size() == order_saved) return;
            order_saved = order.size();
            ids = order;
            for (const std::uint64_t id : previous)
                if (!ordered.count(id)) ids.push_back(id);
        }
        const std::filesystem::path out = OrderPath(), temporary = OrderPath().string() + ".new";
        if (std::FILE* f = std::fopen(temporary.string().c_str(), "wb")) {
            const bool ok = std::fwrite(ids.data(), sizeof(std::uint64_t), ids.size(), f) == ids.size();
            std::fclose(f);
            std::error_code ec;
            if (ok) std::filesystem::rename(temporary, out, ec);
        }
    }

    void Hold(std::uint64_t id, std::shared_ptr<const Cooked> image) {      // (under lock)
        auto old = held.find(id);
        if (old != held.end()) {
            held_bytes -= old->second.image->data.size();
            lru.erase(old->second.lru);
            held.erase(old);
        }
        lru.push_front(id);
        held_bytes += image->data.size();
        held[id] = Held{std::move(image), lru.begin()};
        while (held_bytes > options.memory_budget && lru.size() > 1) {
            const std::uint64_t last = lru.back();
            lru.pop_back();
            auto found = held.find(last);
            held_bytes -= found->second.image->data.size();
            held.erase(found);
        }
    }

    // What the GPU draws: a lossy format decoded back when simulating.
    std::shared_ptr<const Cooked> ForDrawing(codec::Format format, std::uint32_t w, std::uint32_t h,
                                             std::vector<std::uint8_t> data, std::uint32_t alpha_top) const {
        auto image = std::make_shared<Cooked>();
        image->format = format;
        image->width = w;
        image->height = h;
        image->alpha_top = alpha_top;
        if (options.simulate && format != codec::kRgba8) {
            std::vector<std::uint32_t> rgba;
            if (!codec::Decode(format, data.data(), data.size(), w, h, &rgba)) return nullptr;
            if (alpha_top) UnstretchAlpha(&rgba, alpha_top);
            image->format = codec::kRgba8;
            image->alpha_top = 0;
            image->data.resize(rgba.size() * 4);
            std::memcpy(image->data.data(), rgba.data(), image->data.size());
        } else {
            image->data = std::move(data);
        }
        return image;
    }

    // A record read from the store into memory (the loader's; or the caller's,
    // synchronous). `entry` as the index had it when the read was asked.
    void LoadOne(std::uint64_t id, const Entry& entry) {
        std::vector<std::uint8_t> data(entry.bytes);
        bool ok = false;
        {
            std::lock_guard<std::mutex> hold(read_lock);
            ok = read && std::fseek(read, long(entry.offset), SEEK_SET) == 0 &&
                 std::fread(data.data(), 1, data.size(), read) == data.size() &&
                 Checksum(data.data(), data.size()) == entry.checksum;
        }
        std::shared_ptr<const Cooked> image =
            ok ? ForDrawing(entry.format, entry.width, entry.height, std::move(data), entry.alpha_top) : nullptr;
        std::lock_guard<std::mutex> hold(lock);
        loading.erase(id);
        if (image) {
            Hold(id, std::move(image));
            ++loaded;
        } else {
            index[id].rejected = true;     // a damaged record: drawn decoded from now on
            ++load_failed;
        }
    }

    void LoaderLoop() {
        Background();
        for (;;) {
            std::uint64_t id = 0;
            Entry entry;
            {
                std::unique_lock<std::mutex> hold(lock);
                wake_loader.wait(hold, [&] { return stopping || !loads.empty(); });
                if (stopping) return;
                id = loads.front();
                loads.pop_front();
                entry = index[id];
            }
            LoadOne(id, entry);
        }
    }

    // A texture encoded, judged and stored (the cooker's; or the caller's).
    void CookOne(Job job) {
        const auto began = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> data;
        // A lossy format's alpha stretched to its top (the GS's opaque is 0x80
        // or 0x7F, the middle of a byte): opaque and none then at the ends of
        // the range, which block compression keeps exactly.
        std::uint32_t top = 0;
        std::vector<std::uint32_t> source;
        if (options.format != codec::kRgba8) {
            for (const std::uint32_t texel : job.rgba) top = std::max(top, texel >> 24);
            if (top == 0 || top == 255) top = 0;
            if (top) {
                source = job.rgba;
                for (std::uint32_t& texel : source) texel = (texel & 0xFFFFFF) | Stretch(texel >> 24, top) << 24;
            }
        }
        const bool doubled = top != 0;
        const std::uint32_t* encoded = doubled ? source.data() : job.rgba.data();
        bool ok = codec::Encode(options.format, encoded, job.width, job.height, options.effort, &data);
        if (ok && options.format != codec::kRgba8) {
            // Judged as the GPU will read it: close enough, or drawn decoded.
            std::vector<std::uint32_t> back;
            ok = codec::Decode(options.format, data.data(), data.size(), job.width, job.height, &back);
            if (ok && doubled) UnstretchAlpha(&back, top);
            ok = ok && codec::Compare(job.rgba.data(), back.data(), back.size()).Acceptable();
        }
        cook_ns += std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - began).count());
        RecordHeader record{};
        record.magic = kRecordMagic;
        record.flags = ok ? top << 8 : kRejected;
        record.id = job.id;
        record.format = options.format;
        record.width = job.width;
        record.height = job.height;
        record.bytes = ok ? std::uint32_t(data.size()) : 0;
        record.checksum = ok ? Checksum(data.data(), data.size()) : 0;
        Entry entry;
        entry.bytes = record.bytes;
        entry.width = job.width;
        entry.height = job.height;
        entry.checksum = record.checksum;
        entry.format = options.format;
        entry.rejected = !ok;
        entry.alpha_top = ok ? top : 0;
        bool written = false;
        {
            std::lock_guard<std::mutex> hold(append_lock);
            if (append && store_bytes + sizeof record + record.bytes <= options.disk_budget) {
                const long at = std::ftell(append);
                written = at >= 0 && std::fwrite(&record, sizeof record, 1, append) == 1 &&
                          (!record.bytes || std::fwrite(data.data(), 1, record.bytes, append) == record.bytes) &&
                          std::fflush(append) == 0;
                entry.offset = std::uint64_t(at) + sizeof record;
                if (written) store_bytes += sizeof record + record.bytes;
            }
        }
        std::shared_ptr<const Cooked> image =
            ok ? ForDrawing(options.format, job.width, job.height, std::move(data), top) : nullptr;
        std::lock_guard<std::mutex> hold(lock);
        cooking.erase(job.id);
        // Known from the store from now on - or, not written (the store full,
        // no file), for this run only: a refusal remembered, an image held.
        if (written || entry.rejected) index[job.id] = entry;
        if (image) {
            Hold(job.id, std::move(image));
            ++cooked;
            cooked_in += job.rgba.size() * 4;
            cooked_out += record.bytes;
        } else {
            ++rejected;
        }
    }

    void CookerLoop() {
        Background();
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> hold(lock);
                wake_cooker.wait(hold, [&] { return stopping || !cooks.empty(); });
                if (stopping) return;
                job = std::move(cooks.front());
                cooks.pop_front();
                queued_bytes -= job.rgba.size() * 4;
            }
            CookOne(std::move(job));
        }
    }

    void OpenStore() {
        std::error_code ec;
        std::filesystem::create_directories(options.directory, ec);
        path = options.directory / (std::string("textures.") +
                                    (options.format == codec::kAstc4x4 ? "astc"
                                     : options.format == codec::kBc7  ? "bc7"
                                                                      : "rgba8") +
                                    ".store");
        FileHeader expected{};
        std::memcpy(expected.magic, kMagic, 8);
        expected.format = options.format;
        expected.stamp = kCookVersion << 8 | std::uint32_t(options.effort & 0xFF);
        std::uint64_t good_end = 0;
        if (std::FILE* f = std::fopen(path.string().c_str(), "rb")) {
            FileHeader header{};
            if (std::fread(&header, sizeof header, 1, f) == 1 && std::memcmp(&header, &expected, sizeof header) == 0) {
                good_end = sizeof header;
                RecordHeader record{};
                while (std::fread(&record, sizeof record, 1, f) == 1 && record.magic == kRecordMagic) {
                    Entry entry;
                    entry.offset = good_end + sizeof record;
                    entry.bytes = record.bytes;
                    entry.width = record.width;
                    entry.height = record.height;
                    entry.checksum = record.checksum;
                    entry.format = codec::Format(record.format);
                    entry.rejected = (record.flags & kRejected) != 0;
                    entry.alpha_top = (record.flags >> 8) & 0xFF;
                    if (record.bytes && std::fseek(f, long(record.bytes), SEEK_CUR) != 0) break;
                    // A record the file ends inside of (a run ended while it was written).
                    if (std::uint64_t(std::ftell(f)) != entry.offset + record.bytes) break;
                    index[record.id] = entry;
                    good_end = entry.offset + record.bytes;
                }
            }
            std::fclose(f);
        }
        if (!good_end) {
            // A new store (none, or another kind's: put aside).
            if (std::filesystem::exists(path, ec)) std::filesystem::rename(path, path.string() + ".old", ec);
            index.clear();
            if (std::FILE* f = std::fopen(path.string().c_str(), "wb")) {
                std::fwrite(&expected, sizeof expected, 1, f);
                std::fclose(f);
                good_end = sizeof expected;
            }
        } else if (std::filesystem::file_size(path, ec) != good_end) {
            std::filesystem::resize_file(path, good_end, ec);   // the unfinished record dropped
        }
        store_bytes = good_end;
        append = std::fopen(path.string().c_str(), "ab");
        read = std::fopen(path.string().c_str(), "rb");
    }
};

Cache& TheCache() {
    static Cache cache;
    return cache;
}

}  // namespace

void Open(const Options& options) {
    Cache& c = TheCache();
    if (c.open) return;
    c.options = options;
    c.OpenStore();
    c.LoadOrder();
    c.open = true;
    c.loader = std::thread([&c] { c.LoaderLoop(); });
    c.cooker = std::thread([&c] { c.CookerLoop(); });
    // Its threads stopped before the cache itself goes at exit (a thread still
    // joinable as its std::thread is destroyed would end the program).
    static bool registered = false;
    if (!registered) {
        registered = true;
        std::atexit([] { Close(); });
    }
    std::size_t kept = 0, rejected = 0;
    for (const auto& [id, e] : c.index) (e.rejected ? rejected : kept)++;
    std::fprintf(stderr, "[assets] the store %s: %zu textures cooked to %s (%zu drawn decoded), %llu MB%s%s%s\n",
                 c.path.string().c_str(), kept, codec::Name(options.format), rejected,
                 static_cast<unsigned long long>(c.store_bytes.load() >> 20), options.simulate ? "; simulated" : "",
                 options.cook ? "" : "; not cooking", options.synchronous ? "; synchronous" : "");
}

bool Enabled() { return TheCache().open; }
const Options& Settings() { return TheCache().options; }

std::shared_ptr<const Cooked> Find(std::uint64_t id, State* state) {
    Cache& c = TheCache();
    std::unique_lock<std::mutex> hold(c.lock);
    if (c.ordered.insert(id).second) {
        c.order.push_back(id);
        c.Prefetch(id);
    }
    auto in = c.held.find(id);
    if (in != c.held.end()) {
        c.lru.splice(c.lru.begin(), c.lru, in->second.lru);
        *state = State::Ready;
        ++c.found;
        return in->second.image;
    }
    auto known = c.index.find(id);
    if (known != c.index.end()) {
        if (known->second.rejected) {
            *state = State::Rejected;
        } else if (c.options.synchronous) {
            const Entry entry = known->second;
            c.loading.insert(id);
            hold.unlock();
            c.LoadOne(id, entry);
            return Find(id, state);
        } else {
            if (c.loading.insert(id).second) {
                c.loads.push_back(id);
                c.wake_loader.notify_one();
            }
            *state = State::Loading;
        }
        return nullptr;
    }
    if (c.cooking.count(id)) *state = State::Cooking;
    else if (!c.options.cook || c.store_bytes >= c.options.disk_budget) *state = State::Rejected;
    else *state = State::Unknown;
    return nullptr;
}

void Cook(std::uint64_t id, std::vector<std::uint32_t> rgba, std::uint32_t width, std::uint32_t height) {
    Cache& c = TheCache();
    std::unique_lock<std::mutex> hold(c.lock);
    if (!c.open || c.index.count(id) || c.cooking.count(id) || c.held.count(id)) return;
    if (c.options.synchronous) {
        c.cooking.insert(id);
        hold.unlock();
        c.CookOne(Job{id, std::move(rgba), width, height});
        return;
    }
    // Seen faster than cooked (a first race): the rest waits for its next decode.
    if (c.queued_bytes + rgba.size() * 4 > (64ull << 20)) {
        ++c.dropped;
        return;
    }
    c.queued_bytes += rgba.size() * 4;
    c.cooking.insert(id);
    c.cooks.push_back(Job{id, std::move(rgba), width, height});
    c.wake_cooker.notify_one();
}

std::string Report() {
    Cache& c = TheCache();
    if (!c.open) return "";
    static std::uint64_t last[10] = {};
    const std::uint64_t now[10] = {c.cooked, c.rejected, c.cooked_in, c.cooked_out, c.cook_ns,
                                   c.loaded, c.load_failed, c.found, c.dropped, c.prefetched};
    std::uint64_t d[10];
    for (int i = 0; i < 10; ++i) d[i] = now[i] - last[i], last[i] = now[i];
    std::size_t entries = 0, queue = 0;
    std::uint64_t store = 0, held = 0;
    {
        std::lock_guard<std::mutex> hold(c.lock);
        entries = c.index.size();
        store = c.store_bytes;
        held = c.held_bytes;
        queue = c.cooks.size();
    }
    char text[400];
    std::snprintf(text, sizeof text,
                  "assets (%s): %llu cooked (%.1f MB to %.1f MB, %llu not close enough, %.0f ms), %llu waiting, "
                  "%llu left for later; %llu read from the store (%llu ahead of need, %llu bad), %llu used cooked; "
                  "the store %zu textures, %llu MB; %llu MB in memory",
                  codec::Name(c.options.format), (unsigned long long)d[0], double(d[2]) / 1048576.0,
                  double(d[3]) / 1048576.0, (unsigned long long)d[1], double(d[4]) / 1e6, (unsigned long long)queue,
                  (unsigned long long)d[8], (unsigned long long)d[5], (unsigned long long)d[9], (unsigned long long)d[6],
                  (unsigned long long)d[7], entries, (unsigned long long)(store >> 20),
                  (unsigned long long)(held >> 20));
    return text;
}

void Save() {
    Cache& c = TheCache();
    if (c.open) c.SaveOrder();
}

void Close() {
    Cache& c = TheCache();
    if (!c.open) return;
    c.SaveOrder();
    {
        std::lock_guard<std::mutex> hold(c.lock);
        c.stopping = true;
    }
    c.wake_loader.notify_all();
    c.wake_cooker.notify_all();
    if (c.loader.joinable()) c.loader.join();
    if (c.cooker.joinable()) c.cooker.join();
    if (c.append) std::fclose(c.append);
    if (c.read) std::fclose(c.read);
    c.append = c.read = nullptr;
    c.open = false;
    c.stopping = false;
    c.index.clear();
    c.held.clear();
    c.lru.clear();
    c.held_bytes = 0;
    c.loads.clear();
    c.loading.clear();
    c.cooks.clear();
    c.cooking.clear();
    c.queued_bytes = 0;
    c.previous.clear();
    c.previous_at.clear();
    c.order.clear();
    c.ordered.clear();
    c.order_saved = 0;
}

}  // namespace ps2host::assets
