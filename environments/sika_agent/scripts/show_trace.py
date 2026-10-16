"""Print one hosted trace as a readable conversation.

    python3 scripts/show_trace.py <trace-id>

Fetches the stored trace with the `prime` CLI (nothing is written locally) and prints the
request, every assistant message and tool call, tool results, and the verdict metrics.
"""

import json
import subprocess
import sys


def text(content) -> str:
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def main(trace_id: str) -> None:
    cmd = ["prime", "traces", "get", trace_id, "--raw", "--plain"]
    trace = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    data = trace["task"]["data"]
    print(f"task: {data.get('id')}\nexpect: {json.dumps(data.get('expect'))}\n")
    for node in trace["nodes"]:
        message = node.get("message", node)
        role = message.get("role")
        if role == "system":
            continue
        body = text(message.get("content")).strip().replace("\n", " ")
        print(f"[{role}] {body}")
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            print(f"    -> {fn.get('name')}({fn.get('arguments')})")
    print(f"\nmetrics: {trace.get('metrics')}\nrewards: {trace.get('rewards')}")


if __name__ == "__main__":
    main(sys.argv[1])
