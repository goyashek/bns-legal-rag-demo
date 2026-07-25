# Prompts

Keeping all the LLM prompts here as separate files instead of inlining them in the node
code, so they're easier to tweak and diff without touching logic. One file per LLM
node, loaded by the matching `src/agent/nodes/*.py`.

Prompt files (✅ = written):

| File | Node | Model | Job |
|---|---|---|---|
| ✅ `router.txt` | router | legal-easy | 3-way classify: criminal / out_of_scope / needs_clarification |
| ✅ `intent_expander.txt` | intent_expander | legal-easy | narrative to 3-5 offence sub-queries |
| ✅ `grader.txt` | grader | legal-easy | per-chunk relevance yes/no |
| ✅ `rewriter.txt` | rewriter | legal-easy | HyDE / targeted rewrite (by reason) |
| ✅ `generator.txt` | generator | legal-hard | cited LegalAdvice, cite only from provided chunks |
| ✅ `checker.txt` | checker | legal-easy | claim-vs-source faithfulness judge |

All prompts are loaded via `src/agent/llm.py::load_prompt(name)` (reads `<name>.txt`,
uncached so tuning shows up in eval). The `{query}` / context placeholders get
`.format(...)`-spliced at call time.

Conventions I'm sticking to:
- Keep the system/instruction text here; splice in the retrieved context at call time.
- The generator prompt has to forbid citing sections that aren't in the provided chunks.
  The deterministic validator catches it anyway, but if the prompt is clear the violations
  should be rare in the first place.
- Version the prompts in git and jot down any meaningful change in the eval log, otherwise
  I can't tell whether a metric moved because of the prompt or something else.
