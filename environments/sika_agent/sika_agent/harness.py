"""The default harness for this taskset: a plain tool-calling chat loop over the wallet
tools, with no shell or file tools. Exporting it makes `vf-eval sika-agent` use it unless
another harness is chosen with `--env.agent.harness.id`."""

from verifiers.v1.harnesses.null.harness import NullHarness


class SikaHarness(NullHarness):
    pass
