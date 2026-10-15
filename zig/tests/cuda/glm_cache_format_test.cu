#include "../../kernels/cuda/glm_cache.cuh"
#include <cmath>
#include <cstdio>

int main() {
  int checks = 0;
  auto equal = [&](double got, double want, const char* name) {
    ++checks;
    if (got == want) return true;
    std::fprintf(stderr, "%s: got %.17g, want %.17g\n", name, got, want);
    return false;
  };
  bool ok = true;
  ok &= equal(glm_cache::scale_bits(0), 0x00800000u, "zero row scale");
  ok &= equal(glm_cache::scale_bits(0x3f800000u), 0x3b800000u, "unit row scale");
  ok &= equal(glm_cache::scale_bits(0x3fc00000u), 0x3c000000u, "non-power-of-two maximum");
  ok &= equal(glm_cache::scale_bits(0x43800000u), 0x3f800000u, "256 row scale");
  ok &= equal(glm_cache::scale_bits(0x00800000u), 0x00800000u, "minimum scale clamp");
  ok &= equal(glm_cache::encode(256.0f), 0x78, "unit row code");
  ok &= equal(glm_cache::encode(-256.0f), 0xf8, "negative unit row code");
  ok &= equal(glm_cache::encode(128.0f), 0x70, "half row code");
  ok &= equal(glm_cache::encode(0.0f), 0, "zero code");
  ok &= equal(glm_cache::encode(1.0625f), 0x38, "even lower tie");
  ok &= equal(glm_cache::encode(1.1875f), 0x3a, "even upper tie");
  ok &= equal(glm_cache::encode(std::ldexp(1.0f, -7)), 4, "subnormal code");
  for (unsigned b = 0; b < 256; ++b) {
    if ((b & 127) == 127) continue;
    const int exponent = (b >> 3) & 15;
    const float mantissa = b & 7;
    const float magnitude = exponent == 0 ? std::ldexp(mantissa, -9)
                                          : std::ldexp(1.0f + mantissa / 8, exponent - 7);
    ok &= equal(glm_cache::decode(b), b & 128 ? -magnitude : magnitude, "E4M3 decode");
  }
  std::printf("GLM cache format: %d CPU checks, %s\n", checks, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
