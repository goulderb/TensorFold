# Exact current host scan and copy boundary

The retained [bundle](upstream-retained-bundle-20261019.md) reaches about
118.5 subsequent tokens/s at its host-greedy boundary. Before prioritizing
device selection, this CPU seam extracts the current BF16 argmax verbatim
from source a1240bb and uses all 128 distinct consumed real logit rows.
The historical native adapter's 310 us draw belongs to different code.

ReleaseSafe hot-session copy plus scan takes 27.9358 ms across 128 draws,
median of five repetitions, range 27.7259–28.2989 ms, or 218.248 us/draw.
Each changing row is staged into the same session-sized buffer outside
timing. Scan alone is 197.247 us/draw and copy alone 20.631 us/draw; these
separate medians need not sum to the combined median. Rotating 64 MB source
rows measure 227.128 us/draw. ReleaseFast's hot combined 217.215 us/draw
overlaps ReleaseSafe, so safety checks expose no large difference here.

Both builds validate all 128 recorded tokens and every copied byte. They
each pass two semantic tests for ties, signed zero and nonfinite rejection.
Compiler barriers and post-timing hashes preserve output consumption.
The timer baseline is 2.2 us across 128 intervals. The 24-processor host
snapshot reports 11% load; there is no affinity or machine-wide idle claim.

This CPU estimate numerically equals 2.608% of the bundle's decode interval
and 2.370% of its whole turn. CPU staging differs from real D2H cache
ownership and interleaving, so these are fixture attribution estimates,
not a measured removable share or live speedup. Device selection still
needs its reduction, finite-status contract, tiny readback and per-session
logit ownership. Keep it as a small compatible candidate; the screen alone
does not justify a greater-than-5% full-model forecast.

The [archive](evidence/upstream-host-scan-20261019/README.txt) contains 24
members, 31,375 compressed bytes, SHA256
affa16f8061f896d58d20e35b67241991bb16a0ba37ab4a8d49622308172e860.
The original 63,569,920-byte input and executables remain externally pinned.
The standard archive verifier passed. All recorded build, semantic-test and
benchmark processes ended with exit zero. No GPU or serving change occurred.
