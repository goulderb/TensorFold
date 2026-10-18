Generated with the independent, standard-library-only encoder `C:/Development/prep/glm53-fp4-local-20261016/csf-fixtures-sol/generate.py`, derived from `docs/csf-scale-blocks.md` at pinned lil-flashinfer commit `7e203806a82b588b9113f5a0866a028564847741`. The required venv lacks NumPy, so the task's independent-encoder fallback was used. These are synthetic matrices at the real GLM scale geometries, not checkpoint extracts or reference-encoder output.

Reproduce from a different working directory with `C:/Development/local-ai-server/runtime/proxy-venv/Scripts/python.exe -I C:/Development/prep/glm53-fp4-local-20261016/csf-fixtures-sol/generate.py FRESH_OUTPUT_DIRECTORY`. The generator refuses an existing output directory. It does not import or modify the pinned reference.

Envelope format: seven little-endian U32 words contain R, C, and byte lengths of bases, codes, page_bases, offsets, exceptions. Remaining packets contain a little-endian U32 run length and one byte. Expand packets and split the resulting bytes at the five component lengths before decoding. RLE keeps both full-size GLM fixtures below the 200 KB combined budget; each raw code component alone is 262144 bytes.

Source byte at (r,c) is 255 when c=C-1 and r%97=0, otherwise `(r%15)*16 + (15 if c%2 else 0)`. The encoder chooses the smallest base maximizing coverage of a 16-byte window. Expected output is checked per byte against that formula. Additional Zig tests use unrelated random inputs and hand-built noncanonical palettes and exception streams.

SHA256 of fixture envelopes:

```
4044653048eb244adabd755f25249c76224ecb2995450829469b5be391dfe11a  2048x256.rle
ea172cc8996235ecae3c1de386c8b3d054ce6f036acc3ab13d0615f85d47c33e  4096x128.rle
4e4fd0c7e95a0707c1bf5de39ec3f46d100040039ffb5ff899a3ecd990e0a690  32x1.rle
3103a6e93c4490c76ad509255056da0cdcbe8e5b67a3b7f0264cd1b98116adfa  160x5.rle
71f9e0e964ff6184de34c2fade59ad02548ad2fd8174384a6cb0b698df5e5e74  160x3.rle
```

Generator SHA256: `da62abae5eb7aa8c588e393892568556f38e329227ddce80948acc043af9fdf4`.
