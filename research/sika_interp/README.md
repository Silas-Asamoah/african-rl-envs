# sika-interp

Looks inside open models while they work through [sika-agent](../../environments/sika_agent)
tasks, comparing requests in English, Twi and Twi–English. The models are Qwen3.5-9B and
Ministral 3 8B Reasoning, both run on our own GPUs. Work in progress: no results yet.

## What's here

- `sika_interp/tasks.py`: the study's episodes. Matched twins (same wallet, different
  wording) in all three languages for 10 seeds per family, plus 10 more Twi–English seeds per
  family.
- `sika_interp/words.py`: tags each word of a Twi–English request as English, Twi, mixed (an
  English stem in Twi morphology, like "a-send-i") or other (names, brands, amounts), and
  finds the switch points between them. The tags are heuristics checked against the
  single-language wording pools; a native speaker should spot-check them before analysis
  relies on them.

## Development

```bash
uv sync
uv run pytest -q
```

Run outputs (activations, neuron maps, episodes, rendered pages) go to `acts/`, `maps/`,
`runs/` and `outputs/`, which git ignores.
