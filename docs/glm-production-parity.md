# GLM production execution parity

The native GLM candidate must match the production execution profile before
another round of kernel tuning. The comparison includes native FP4 Local as
the serving target and the same routed-only EXL3 checkpoint on both engines.
Checkpoint differences cannot explain an implementation result in that control.

The selected design uses the second architecture candidate. It keeps deferred
draft shape in the lane contract and one owned pending pass in GLM. Resolution
happens only when a queued neural head wins over a forced or copied continuation,
before allocating verification rows. Existing fixed-draft depth planning already
skips neural work for copy winners. Preserve that behavior and its proposer
accounting rather than introducing a second copy preflight.

The transport borrows the existing communicator through `cuda.tp_group.Group`.
Its optional RoCE resources have a shorter lifetime than the communicator.
Control and cache protocols retain their existing NCCL operations. Collective
order must also hold across graph replay and prompt side streams.

Q4 DSA uses one combined QA and KVA projection, followed by two strided norms.
The loader retains separate source-format and full-EXL3 projections, and the
BF16 recovery layout. Q4 input layout has a new numerical identity so saved
states from the earlier two-projection implementation cannot be reused silently.
Startup receipts report the loaded projection formats rather than only flags.

L2 prefetch follows the production attention, FFN, and output sites, including
shared-expert computation before routed experts on the enabled decode path.
The planner owns immutable bounded weight spans. Its side stream and events
outlive every graph that references them, and every forward joins the stream.

Verification proceeds through host dispatch and scheduling tests, bounded paired
GPU proofs, then an unprofiled matched HTTP comparison. Device proofs cover Q4
numerics, deferred draft invalidation, RoCE graph replay, and prefetch event
ordering. Whole-model checks include tools, vision, cancellation, occupied
capacity, cache eviction, first restore after both restart types, and a soak.
Local-attention prefill remains disabled because its previous first-restore
failure has no engine repair. Passing a source build or a component proof does
not establish production parity or authorize a promotion verdict.
