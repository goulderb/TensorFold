/* tfclef: native Clef Flash decisions on CUDA (TensorFold), C ABI version 1.
 *
 * One engine per process owns one CUDA context. Calls are serialized by the caller; a concurrent call returns
 * TF_CLEF_BUSY without touching the engine. The joint schema head reimplements Cloudflare's JointSchemaHead
 * (clef-flash joint_schema_model.py, Apache-2.0, revision fde727a287004204b7518dcc983fe64379776712); no model
 * code is copied. See NOTICE and LICENSES/donors/README.md.
 */
#ifndef TFCLEF_H
#define TFCLEF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TF_CLEF_ABI_VERSION 1u

typedef struct tf_clef_engine tf_clef_engine;

typedef enum {
    TF_CLEF_OK = 0,
    TF_CLEF_INVALID = 1,  /* the call or record failed validation; the engine is unchanged */
    TF_CLEF_CAPACITY = 2, /* n_ids exceeds max_tokens; the engine is unchanged */
    TF_CLEF_DEVICE = 3,   /* a device fault the engine drained and reset from; it stays usable */
    TF_CLEF_FATAL = 4,    /* a fault the engine could not drain; every later call returns FATAL; close and exit */
    TF_CLEF_BUSY = 5,     /* another call is in progress */
    TF_CLEF_NONFINITE = 6 /* a logit was NaN or infinite; the engine stays usable */
} tf_clef_status;

enum { TF_CLEF_TYPE_NOUL = 0, TF_CLEF_TYPE_CHOICE = 1, TF_CLEF_TYPE_SCORE = 2 };

/* Parity runs: the head weights and the output table only, with no backbone. */
enum { TF_CLEF_OPEN_HEAD_ONLY = 1u };

/* [start, end): nonempty, end <= n_ids. */
typedef struct {
    uint32_t start, end;
} tf_clef_span;

typedef struct {
    uint8_t type;          /* TF_CLEF_TYPE_*: the official QUESTION_TYPES value */
    tf_clef_span question; /* EncodedQuestion.question_span */
    uint32_t first_option; /* index into options; the questions' option ranges tile options in order */
    uint32_t option_count; /* at least 1 */
} tf_clef_question;

typedef struct {
    const uint32_t *ids; /* EncodedRecord.input_ids */
    uint32_t n_ids;
    const tf_clef_question *questions; /* record order */
    uint32_t n_questions;
    const tf_clef_span *options; /* every question's option_spans, flattened in question order */
    uint32_t n_options;
} tf_clef_record;

typedef struct {
    const char *model_dir;   /* UTF-8, NUL-terminated: the pinned checkpoint directory */
    uint8_t head_sha256[32]; /* joint_head.safetensors, from the verified Hugging Face metadata */
    uint8_t device_uuid[16]; /* must equal the device the engine opens */
    uint64_t cap_bytes;      /* refuse to open when the budget exceeds this or the device's free memory */
    uint32_t max_tokens;     /* 2..4096 */
    uint32_t flags;          /* TF_CLEF_OPEN_* */
} tf_clef_open_options;

typedef struct {
    uint32_t abi_version;
    uint32_t max_tokens, max_questions, max_options;
    uint64_t budget_bytes, device_free_bytes, device_total_bytes;
    uint8_t device_uuid[16];
    char source_commit[41]; /* the TensorFold commit the library was built from, NUL-terminated */
    uint8_t head_only;
} tf_clef_info;

uint32_t tf_clef_abi_version(void);

/* NULL on failure, with a NUL-terminated message truncated to err_len in err. */
tf_clef_engine *tf_clef_open(const tf_clef_open_options *options, char *err, size_t err_len);

/* Backbone and head over one record. logits: one BF16 value widened to float per option, in record option order;
 * cap >= n_options. */
tf_clef_status tf_clef_decide(tf_clef_engine *engine, const tf_clef_record *record, float *logits, uint32_t cap);

/* The head alone over caller hidden rows: n_ids x 4096 BF16 bits, the backbone's rows after its final norm, in host
 * memory. The same record type and head code as tf_clef_decide. */
tf_clef_status tf_clef_head(tf_clef_engine *engine, const tf_clef_record *record, const uint16_t *hidden_bf16,
                            float *logits, uint32_t cap);

tf_clef_status tf_clef_info_get(tf_clef_engine *engine, tf_clef_info *out);

/* The message of the last non-OK status, NUL-terminated and truncated to len; returns its full length. */
size_t tf_clef_last_error(tf_clef_engine *engine, char *buf, size_t len);

/* Drains and frees; a poisoned engine releases host state only. NULL is a no-op; a closed engine is gone. */
void tf_clef_close(tf_clef_engine *engine);

#ifdef __cplusplus
}
#endif

#endif
