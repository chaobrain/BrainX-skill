# Hand off a fixed refinement snapshot

Run documentation maintenance in the authoring checkout, before launching a
new isolated evaluator. Do not let an evaluation run fetch and rewrite its
installed skills or discover older case outputs to repair its own instructions.

1. Finish reviewable edits and focused checks in the authoring checkout.
2. Freeze the skill snapshot and exact execution environment using
   `how-to-refine-skill.md` and the repository's maintained harness. Record the
   declared BrainX tuple, Python, CLI, immutable image ID, prompt fingerprint,
   and skill snapshot hash in the applicable receipts.
3. Compare numbered runs only when the prompt, image, provider/model, and
   execution conditions match and the intended skill edit is the only change.
   If the release, interpreter, or image changes, establish a new baseline.

For a harness-driven pipeline, call this skill in a separate maintenance phase,
then snapshot its result, then launch the read-only evaluation. An interrupted
or rate-limited evaluation remains incomplete; do not attribute it to a
documentation change or use it as evidence of improvement.
