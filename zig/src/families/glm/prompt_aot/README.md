# Embedded GLM prompt latent kernels

Original work: Ash Hart and TensorFold contributors, Apache-2.0; retained LICENSE and original latent.py/kv8.py accompany the unmodified captured cubins, PTX and Triton3.7.1 metadata. Root NOTICE and LICENSES/donors retain the fork and transitive donor credits.

Captured from the immutable production image recorded in provenance.json. These are the H32 sparse onepass and generic-R65 BF16 absorb/expand variants. No CUDA/Triton body was edited. Local cuda_prompt_latent_aot.zig embeds the exact bytes and adds startup integrity checks, fixed prompt eligibility and typed raw-ABI adaptation through the existing cuda.triton.Kernel. It does not authorize publication or change upstream licensing.

provenance.json and retained capture receipts record source/image/variant provenance. The single canonical embedded inventory and numerical/cache identity live in cuda_prompt_latent_aot.zig; no external runtime asset directory is required.
