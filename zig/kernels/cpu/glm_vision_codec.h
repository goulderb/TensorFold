#ifndef TF_GLM_VISION_CODEC_H
#define TF_GLM_VISION_CODEC_H
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define TF_GLM_CODEC_EXPORT __declspec(dllexport)
#else
#define TF_GLM_CODEC_EXPORT __attribute__((visibility("default")))
#endif

#define TF_GLM_CODEC_ABI 1u
#define TF_GLM_CODEC_PNG 1u
#define TF_GLM_CODEC_JPEG 2u
#define TF_GLM_CODEC_WEBP 4u
#define TF_GLM_CODEC_OK 0
#define TF_GLM_CODEC_LIBRARY 1
#define TF_GLM_CODEC_SYMBOL 2
#define TF_GLM_CODEC_VERSION 3
#define TF_GLM_CODEC_UNAVAILABLE 4
#define TF_GLM_CODEC_INVALID 5
#define TF_GLM_CODEC_LIMIT 6
#define TF_GLM_CODEC_COLOR 7
#define TF_GLM_CODEC_MEMORY 8
#define TF_GLM_CODEC_ANIMATED 9

/* Codec structs and longjmp frames never cross this application-owned ABI.
 * Every non-null library path is explicit and absolute. Missing paths disable
 * that codec; a requested library or symbol failing to load fails open().
 * JPEG/WebP support requires genuine headers at C compilation time. */
TF_GLM_CODEC_EXPORT uint32_t tf_glm_codec_abi(void);
TF_GLM_CODEC_EXPORT void *tf_glm_codec_open(const char *png_path,
    const char *jpeg_path, const char *webp_path,
    uint32_t png_version, uint32_t jpeg_abi, uint32_t webp_version, int *status);
TF_GLM_CODEC_EXPORT void tf_glm_codec_close(void *context);
TF_GLM_CODEC_EXPORT uint32_t tf_glm_codec_features(const void *context);
TF_GLM_CODEC_EXPORT int tf_glm_codec_probe(void *context, uint32_t format,
    const uint8_t *encoded, size_t encoded_size, uint32_t max_dimension,
    size_t max_pixels, uint32_t *width, uint32_t *height);
TF_GLM_CODEC_EXPORT int tf_glm_codec_decode(void *context, uint32_t format,
    const uint8_t *encoded, size_t encoded_size, uint32_t width, uint32_t height,
    uint8_t *rgba, size_t rgba_size);
#endif
