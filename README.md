# African RL environments

Reinforcement learning environments built around how people in West Africa actually live and
pay, written for Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
and the Environments Hub.

| Environment | What it trains | Status |
|---|---|---|
| [`sika-agent`](environments/sika_agent) | A mobile money and pay-as-you-go credit assistant for Ghana, in English, Twi and Twi–English code-switching, scored on the final wallet state with penalties for harmful actions | v0.1 |

## Working on an environment

```bash
cd environments/sika_agent
uv sync
uv run pytest -q
uv run vf-eval sika-agent -n 51 -r 1 -m <model>
```
