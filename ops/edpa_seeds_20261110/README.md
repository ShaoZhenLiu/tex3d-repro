# Cloth-adapted EDPA fixed-seed follow-up

The fixed set is 42, 7, 66 and 88. Each EDPA texture receives 5,000 batch-eight source updates and one final export. The 296-frame pool, parameterization, bounds, step size, shared UV sequence and correct spatial-TV definition match the existing source recipe. EDPA's projector-token contrastive/alignment loss is executed from the existing production file's AST, including its EMA normalization. No target inference enters source optimization.

The existing Native/Matched records and Plain controls use task initial states 10--19. Their exact query-indexed-noise protocol is retained: Spatial, Object and Goal run on 41; Long runs on 09. Source GPUs are restricted to 0--3 on both hosts, sharing only with sufficient VRAM and host resources. The already-authorized frozen renderer exception is retained and measured.

Inputs and all baseline episode checksums are frozen before launch. Interrupted source workers resume the texture, EMA and exact replay position. Resource pauses never replace a seed or count as robot failures. Report every fixed seed, four-suite macro-average G, and the sample standard deviation over seeds. The anticipated method ordering is not an acceptance gate. This follow-up does not turn the previously selected reference cases into a random sample of the earlier eighteen runs.
