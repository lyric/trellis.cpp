#include "shape_checkpoint.h"

#include <cstdio>
#include <string>

int main() {
    const std::string path = "shape_checkpoint_test.bin";
    trellis::ShapeCheckpoint expected;
    expected.seed = 42;
    expected.resolution = 512;
    expected.gss = 7.0f;
    expected.gsh = 6.5f;
    expected.max_tokens = 4096;
    expected.coords = {{1, 2, 3}, {4, 5, 6}};
    expected.features.resize(64);
    for (size_t i = 0; i < expected.features.size(); ++i) expected.features[i] = (float)i / 10.0f;

    std::string error;
    if (!trellis::save_shape_checkpoint(path, expected, error)) {
        fprintf(stderr, "save failed: %s\n", error.c_str());
        return 1;
    }

    trellis::ShapeCheckpoint actual;
    if (!trellis::load_shape_checkpoint(path, actual, error)) {
        fprintf(stderr, "load failed: %s\n", error.c_str());
        std::remove(path.c_str());
        return 1;
    }

    if (actual.seed != expected.seed || actual.resolution != expected.resolution ||
        actual.cascade != expected.cascade || actual.gss != expected.gss || actual.gsh != expected.gsh ||
        actual.max_tokens != expected.max_tokens || actual.coords != expected.coords || actual.features != expected.features) {
        fprintf(stderr, "checkpoint round trip did not preserve state\n");
        std::remove(path.c_str());
        return 1;
    }

    FILE* truncated = fopen(path.c_str(), "wb");
    fwrite("TRLSCHK1", 1, 8, truncated);
    fclose(truncated);
    if (trellis::load_shape_checkpoint(path, actual, error)) {
        fprintf(stderr, "truncated checkpoint was accepted\n");
        std::remove(path.c_str());
        return 1;
    }
    std::remove(path.c_str());

    expected.features.pop_back();
    if (trellis::save_shape_checkpoint(path, expected, error)) {
        fprintf(stderr, "invalid feature count was accepted\n");
        std::remove(path.c_str());
        return 1;
    }

    printf("shape checkpoint tests passed\n");
    return 0;
}
