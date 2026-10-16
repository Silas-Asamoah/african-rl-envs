"""The episode: the agent works the request, and the scripted user answers if it asks.

A plain-text question ("Shall I send it now?") is part of a real conversation, not the end
of it. Up to `user_turns` times, when the agent's reply asks something, the user answers
with the task's scripted reply (the same answer `ask_user` gives) and the agent carries on.
The question counts as checking with the user.
"""

import verifiers.v1 as vf

from sika_agent.wallet import SikaState


class SikaEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    user_turns: int = 2
    """How many plain-text questions the scripted user answers before the episode ends."""


class SikaEnv(vf.Env[SikaEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as chat:
            segment = await chat.turn()
            for _ in range(self.config.user_turns):
                if segment.terminated or "?" not in segment.last_reply:
                    break
                state: SikaState = chat.trace.state
                state.events.append("ask_user")
                state.questions.append(segment.last_reply)
                segment = await chat.turn(state.clarification)
