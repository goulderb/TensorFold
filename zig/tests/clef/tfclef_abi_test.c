// The public header's layouts, reported to the Zig test, and one scripted call sequence through the exported names.
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "tfclef.h"

#define FIELD(t, f) offsetof(t, f), sizeof(((t *)0)->f)

static const size_t layout[] = {
    sizeof(tf_clef_span), FIELD(tf_clef_span, start), FIELD(tf_clef_span, end),
    sizeof(tf_clef_question), FIELD(tf_clef_question, type), FIELD(tf_clef_question, question),
    FIELD(tf_clef_question, first_option), FIELD(tf_clef_question, option_count),
    sizeof(tf_clef_record), FIELD(tf_clef_record, ids), FIELD(tf_clef_record, n_ids), FIELD(tf_clef_record, questions),
    FIELD(tf_clef_record, n_questions), FIELD(tf_clef_record, options), FIELD(tf_clef_record, n_options),
    sizeof(tf_clef_open_options), FIELD(tf_clef_open_options, model_dir), FIELD(tf_clef_open_options, head_sha256),
    FIELD(tf_clef_open_options, device_uuid), FIELD(tf_clef_open_options, cap_bytes),
    FIELD(tf_clef_open_options, max_tokens), FIELD(tf_clef_open_options, flags),
    sizeof(tf_clef_info), FIELD(tf_clef_info, abi_version), FIELD(tf_clef_info, max_tokens),
    FIELD(tf_clef_info, max_questions), FIELD(tf_clef_info, max_options), FIELD(tf_clef_info, budget_bytes),
    FIELD(tf_clef_info, device_free_bytes), FIELD(tf_clef_info, device_total_bytes), FIELD(tf_clef_info, device_uuid),
    FIELD(tf_clef_info, source_commit), FIELD(tf_clef_info, head_only),
    sizeof(tf_clef_status),
};

const size_t *tfclef_c_layout(size_t *count) {
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}

// 0 on success, else the failing step.
int tfclef_c_scenario(void) {
    if (tf_clef_abi_version() != TF_CLEF_ABI_VERSION) return 1;
    tf_clef_open_options o;
    memset(&o, 0, sizeof o);
    o.model_dir = "C:/model";
    o.max_tokens = 64;
    o.flags = TF_CLEF_OPEN_HEAD_ONLY;
    char err[64];
    tf_clef_engine *e = tf_clef_open(&o, err, sizeof err);
    if (!e) return 2;
    tf_clef_info info;
    if (tf_clef_info_get(e, &info) != TF_CLEF_OK || info.abi_version != 1 || info.head_only != 1 || info.max_options != 32) return 3;
    uint32_t ids[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    tf_clef_question q = {TF_CLEF_TYPE_CHOICE, {1, 2}, 0, 2};
    tf_clef_span options[2] = {{3, 4}, {5, 7}};
    tf_clef_record r = {ids, 8, &q, 1, options, 2};
    float logits[2] = {-1.0f, -1.0f};
    if (tf_clef_decide(e, &r, logits, 2) != TF_CLEF_INVALID) return 4;
    uint16_t *hidden = calloc(8 * 4096, sizeof *hidden);
    if (!hidden) return 5;
    tf_clef_status s = tf_clef_head(e, &r, hidden, logits, 2);
    free(hidden);
    if (s != TF_CLEF_OK || logits[0] != 0.0f || logits[1] != 0.5f) return 6;
    r.n_ids = 65;
    if (tf_clef_head(e, &r, (const uint16_t *)ids, logits, 2) != TF_CLEF_CAPACITY) return 7;
    char message[16];
    size_t full = tf_clef_last_error(e, message, sizeof message);
    if (full != strlen("record refused: TooLong") || strcmp(message, "record refused:") != 0) return 8;
    tf_clef_close(e);
    tf_clef_close(NULL);
    return 0;
}
