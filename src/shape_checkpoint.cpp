#include "shape_checkpoint.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#ifdef _WIN32
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace trellis {
namespace {

constexpr char MAGIC[8] = {'T', 'R', 'L', 'S', 'C', 'H', 'K', '1'};
constexpr uint32_t VERSION = 1;
constexpr uint32_t STAGE_SHAPE_SLAT = 1;
constexpr uint32_t CHANNELS = 32;
constexpr uint64_t MAX_COORDS = 10000000;

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t stage;
    uint32_t seed;
    uint32_t resolution;
    uint32_t flags;
    uint32_t channels;
    uint32_t max_tokens;
    float gss;
    float gsh;
    uint64_t coord_count;
};
static_assert(sizeof(Header) == 56, "checkpoint header layout changed");

bool write_all(FILE* file, const void* data, size_t size, std::string& error) {
    if (size == 0 || fwrite(data, 1, size, file) == size) return true;
    error = "could not write checkpoint: " + std::string(std::strerror(errno));
    return false;
}

bool read_all(FILE* file, void* data, size_t size, std::string& error) {
    if (size == 0 || fread(data, 1, size, file) == size) return true;
    error = feof(file) ? "checkpoint is truncated" : "could not read checkpoint: " + std::string(std::strerror(errno));
    return false;
}

}  // namespace

bool save_shape_checkpoint(const std::string& path, const ShapeCheckpoint& checkpoint, std::string& error) {
    if (path.empty()) { error = "checkpoint path is empty"; return false; }
    if (checkpoint.coords.empty()) { error = "checkpoint has no coordinates"; return false; }
    if (checkpoint.coords.size() > MAX_COORDS) { error = "checkpoint has too many coordinates"; return false; }
    if (checkpoint.resolution < 512 || checkpoint.resolution > 1536 || checkpoint.resolution % 128 != 0) {
        error = "checkpoint resolution is invalid";
        return false;
    }
    if (checkpoint.cascade != (checkpoint.resolution > 512)) {
        error = "checkpoint cascade flag does not match its resolution";
        return false;
    }
    if (!std::isfinite(checkpoint.gss) || !std::isfinite(checkpoint.gsh)) {
        error = "checkpoint guidance settings are invalid";
        return false;
    }
    if (checkpoint.max_tokens <= 0 || checkpoint.max_tokens > 10000000) {
        error = "checkpoint token budget is invalid";
        return false;
    }
    if (checkpoint.features.size() != checkpoint.coords.size() * CHANNELS) {
        error = "checkpoint feature count does not match coordinates";
        return false;
    }
    const int grid = checkpoint.resolution / 16;
    for (const auto& coord : checkpoint.coords) {
        if (coord[0] < 0 || coord[0] >= grid || coord[1] < 0 || coord[1] >= grid || coord[2] < 0 || coord[2] >= grid) {
            error = "checkpoint coordinate is outside its resolution grid";
            return false;
        }
    }
    for (float value : checkpoint.features) {
        if (!std::isfinite(value)) { error = "checkpoint contains a non-finite feature"; return false; }
    }

    const std::string temporary = path + ".tmp";
    FILE* file = fopen(temporary.c_str(), "wb");
    if (!file) { error = "could not create checkpoint: " + std::string(std::strerror(errno)); return false; }

    Header header{};
    std::memcpy(header.magic, MAGIC, sizeof(MAGIC));
    header.version = VERSION;
    header.stage = STAGE_SHAPE_SLAT;
    header.seed = checkpoint.seed;
    header.resolution = (uint32_t)checkpoint.resolution;
    header.flags = checkpoint.cascade ? 1u : 0u;
    header.channels = CHANNELS;
    header.max_tokens = (uint32_t)checkpoint.max_tokens;
    header.gss = checkpoint.gss;
    header.gsh = checkpoint.gsh;
    header.coord_count = checkpoint.coords.size();

    bool ok = write_all(file, &header, sizeof(header), error);
    for (const auto& coord : checkpoint.coords) {
        const int32_t xyz[3] = {coord[0], coord[1], coord[2]};
        if (ok) ok = write_all(file, xyz, sizeof(xyz), error);
    }
    if (ok) ok = write_all(file, checkpoint.features.data(), checkpoint.features.size() * sizeof(float), error);
    if (ok && fflush(file) != 0) { error = "could not flush checkpoint: " + std::string(std::strerror(errno)); ok = false; }
#ifdef _WIN32
    if (ok && _commit(_fileno(file)) != 0) { error = "could not sync checkpoint: " + std::string(std::strerror(errno)); ok = false; }
#else
    if (ok && fsync(fileno(file)) != 0) { error = "could not sync checkpoint: " + std::string(std::strerror(errno)); ok = false; }
#endif
    if (fclose(file) != 0 && ok) { error = "could not close checkpoint: " + std::string(std::strerror(errno)); ok = false; }

    if (!ok) { std::remove(temporary.c_str()); return false; }
#ifdef _WIN32
    if (!MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "could not publish checkpoint: Windows error " + std::to_string(GetLastError());
        std::remove(temporary.c_str());
        return false;
    }
#else
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        error = "could not publish checkpoint: " + std::string(std::strerror(errno));
        std::remove(temporary.c_str());
        return false;
    }
#endif
    return true;
}

bool load_shape_checkpoint(const std::string& path, ShapeCheckpoint& checkpoint, std::string& error) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) { error = "could not open checkpoint: " + std::string(std::strerror(errno)); return false; }

    Header header{};
    bool ok = read_all(file, &header, sizeof(header), error);
    if (ok && std::memcmp(header.magic, MAGIC, sizeof(MAGIC)) != 0) { error = "not a Trellis checkpoint"; ok = false; }
    if (ok && header.version != VERSION) { error = "unsupported checkpoint version " + std::to_string(header.version); ok = false; }
    if (ok && header.stage != STAGE_SHAPE_SLAT) { error = "unsupported checkpoint stage"; ok = false; }
    if (ok && header.channels != CHANNELS) { error = "checkpoint has an unsupported channel count"; ok = false; }
    if (ok && (header.flags & ~1u) != 0) { error = "checkpoint has unsupported flags"; ok = false; }
    if (ok && (header.max_tokens == 0 || header.max_tokens > 10000000)) { error = "checkpoint token budget is invalid"; ok = false; }
    if (ok && (header.coord_count == 0 || header.coord_count > MAX_COORDS)) { error = "checkpoint coordinate count is invalid"; ok = false; }
    if (ok && (header.resolution < 512 || header.resolution > 1536 || header.resolution % 128 != 0)) {
        error = "checkpoint resolution is invalid";
        ok = false;
    }
    if (ok && ((header.flags & 1u) != 0) != (header.resolution > 512)) {
        error = "checkpoint cascade flag does not match its resolution";
        ok = false;
    }
    if (ok && (!std::isfinite(header.gss) || !std::isfinite(header.gsh))) {
        error = "checkpoint guidance settings are invalid";
        ok = false;
    }
    if (ok && header.coord_count > std::numeric_limits<size_t>::max() / CHANNELS) {
        error = "checkpoint is too large";
        ok = false;
    }

    ShapeCheckpoint loaded;
    if (ok) {
        loaded.seed = header.seed;
        loaded.resolution = (int)header.resolution;
        loaded.cascade = (header.flags & 1u) != 0;
        loaded.gss = header.gss;
        loaded.gsh = header.gsh;
        loaded.max_tokens = (int)header.max_tokens;
        loaded.coords.resize((size_t)header.coord_count);
        loaded.features.resize((size_t)header.coord_count * CHANNELS);
        for (auto& coord : loaded.coords) {
            int32_t xyz[3];
            if (!read_all(file, xyz, sizeof(xyz), error)) { ok = false; break; }
            const int grid = (int)header.resolution / 16;
            if (xyz[0] < 0 || xyz[0] >= grid || xyz[1] < 0 || xyz[1] >= grid || xyz[2] < 0 || xyz[2] >= grid) {
                error = "checkpoint coordinate is outside its resolution grid";
                ok = false;
                break;
            }
            coord = {xyz[0], xyz[1], xyz[2]};
        }
        if (ok) ok = read_all(file, loaded.features.data(), loaded.features.size() * sizeof(float), error);
        if (ok) {
            for (float value : loaded.features) {
                if (!std::isfinite(value)) { error = "checkpoint contains a non-finite feature"; ok = false; break; }
            }
        }
        if (ok && fgetc(file) != EOF) { error = "checkpoint has unexpected trailing data"; ok = false; }
    }
    fclose(file);
    if (!ok) return false;
    checkpoint = std::move(loaded);
    return true;
}

}  // namespace trellis
