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

- `sika_interp/serving.py`: adds Mistral's reasoning prompt to Ministral's requests, as its
  model card recommends; without it the model doesn't reason. Runs as a proxy in front of
  vLLM, and the internal-measurement jobs use the same function to render prompts.
- `sika_interp/traces.py`: rebuilds each turn's exact chat request from an eval trace.
- `sika_interp/modal_app.py`: loads a model on a Modal GPU, re-renders every turn of the
  given episodes, checks each against the prompt length the server reported, and saves the
  residual stream and MLP neurons at the last prompt token:
  `uvx modal run -m sika_interp.modal_app --traces outputs/<run>/traces.jsonl`.

## Playing episodes on a Prime GPU pod

`remote/pod.sh` serves both models with vLLM and plays episodes with the configs in
`configs/` (each model's recommended sampling, matched tasks on). From a Prime H200 pod
(`ubuntu_22_cuda_12` image) with a copy of the repo:

```bash
bash research/sika_interp/remote/pod.sh setup
bash research/sika_interp/remote/pod.sh serve qwen && bash research/sika_interp/remote/pod.sh wait qwen
bash research/sika_interp/remote/pod.sh serve ministral && bash research/sika_interp/remote/pod.sh wait ministral
bash research/sika_interp/remote/pod.sh eval qwen -n 60
```

The pod's CUDA 12.4 driver needs vLLM's CUDA 12.9 build and prebuilt kernels; the script
takes care of both.

## Development

```bash
uv sync
uv run pytest -q
```

Run outputs (activations, neuron maps, episodes, rendered pages) go to `acts/`, `maps/`,
`runs/` and `outputs/`, which git ignores.
