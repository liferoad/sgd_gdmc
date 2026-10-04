# Publication package

Main draft: `article.md`. Browser preview: `article.html`. Supporting detail: `appendix.md` / `appendix.html`. The article is written in the author's first person, with a concrete question, plain-language explanation, and figures, following the accessible experimental style of the BinaryNets and AI/ML Demystified posts.

## Medium preparation

- Copy the article text into a Medium draft. Upload PNGs 01–04 at the marked positions; retain their captions and alt text. The numbered figure captions start after the conceptual cover.
- Keep figures 05–06 in the companion, or insert them into the supporting-experiments section for a longer post. SVG versions are provided for editing.
- PNGs 07 and 08 are upload-ready image versions of the two numerical tables if the editor does not preserve table formatting. The methods table can be converted to a short list. The main information is also present in figures and prose.
- Replace the two relative `appendix.html` links with the final public companion/repository URL before publication. No public repository URL was assumed. The HTML files and assets work together locally; preserve the folder layout when sharing.
- Suggested tags: Machine Learning, Deep Learning, Optimization, Quantization, Research.
- Alternative titles: “Learning on a Grid: Testing GDMC for Low-Bit Neural Networks”; “Sixteen Weight Values: What GDMC Teaches Us About Low-Bit Training.”

## Scientific decisions in the draft

- Recomputed figures from raw CSVs and used final accuracy as the overview metric. Peak test accuracy is always labeled as descriptive/test-selected.
- Corrected the prior prose claim that momentum dominates basic GDMC: it fails badly at 2 bits and has slightly lower final Fashion 4-bit accuracy.
- Scoped the 4-bit win to the tested projected Adam, highlighting constant learning rates and the shorter validation-tuning horizon.
- Explicitly marked QAT provisional because training/evaluation bias policies differ. This draft reports the existing experiment; it does not silently repair code or invent corrected results.
- Did not call the experiment “powered”: seed counts increased, but no prospective power analysis is available. Did not infer equivalence from a nonsignificant difference.
- Did not call low-bit values packed low-bit training, or claim memory/runtime/energy improvements on large models.
- Supporting memory plot uses total process peak RSS; CPU measurements are single runs. Historical results with known protocol bugs are provenance, not new headline evidence.

## Rebuild and provenance

Run `.venv/bin/python blog/gdmc/build.py` from the repository root. The script checks primary run counts, unique method/bit/seed combinations, curve horizons, and agreement between curve endpoints and raw final scores. `source_manifest.json` includes hashes of every plotted raw source and the repository revision.

The package has not been published or uploaded to Medium. Article and appendix should be reviewed by the author before publication, particularly the framing of the QAT qualification and provisional conclusions.
