# OpenHands A2A agent

The [OpenHands](https://github.com/OpenHands/software-agent-sdk) agent wrapped with the
`agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as
`@local/agentenv-framework/agents/openhands`. This directory is the whole build context: the
protocol package and the OpenHands SDK come from PyPI (`requirements.txt`), and `install.sh` installs
them in a Python of uv's own, with tmux and git for the agent's terminal, both for the image and for `install/v1`.

Instead of wrapping a CLI, the agent drives the OpenHands SDK in process. `openhands_agent.py` is the
A2A side and imports no SDK; `openhands_runtime.py` runs the conversation. Each task is one `arun()`
of an SDK conversation in `/workspace`, with the SDK's default tools (terminal, file editor, task
tracker; no browser). The model is called through the `[model]` LiteLLM endpoint as
`litellm_proxy/<model>`, with `LITELLM_EXTRA_HEADERS` sent on every call.

A context maps to one conversation, whose id is derived from the context id, persisted in
`~/.openhands-a2a/conversations/<id>`, so follow-up tasks in the context resume it, across restarts too.
Images are sent to the model as images; every file part is also saved under `~/.openhands-a2a/inputs`
and its path added to the prompt, for the agent's tools.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `OpenHandsConfig`: `model`, `effort` (the LLM's reasoning effort), `system_prompt` (replaces OpenHands' own), `append_system_prompt`, `max_iterations`, `timeout_seconds` |
| `mcp-config/v1` | SDK default; each server becomes an HTTP `MCPServer` in the agent's `mcp_config`, with its headers |
| `skill-config/v1` | inline and bundle skills installed as `SKILL.md` directories under `~/.openhands-a2a/skills`, loaded into the agent's context |
| `trajectory/v1` | live: the task's OpenHands SDK events, appended as the conversation emits them (format `openhands-events/v1`) |
| `snapshot/v1` | `save` uploads the context's persisted conversation as a gzipped tar and, when asked, the workspace as another; `load` replaces the workspace's contents with the saved ones (extracted aside first, so an archive that fails to extract changes nothing) and installs the conversation under the target context's id |
| `peer-agents/v1` | SDK default: the `peer_agents` MCP server reaches the agent like any other |
| `install/v1` | copies this directory into the task container as `/opt/openhands-a2a`, runs `install.sh` (Debian or Ubuntu, as root, with network access), writes the LiteLLM credentials to a 0600 `agent.env` through a heredoc and starts `start.sh`, which sources it |
| `triggers/v1` | SDK default |

`OPENHANDS_A2A_MODEL` sets the model when agent-config sets none, as for a peer that other agents
message directly, and `OPENHANDS_A2A_WORKSPACE` the workspace (default `/workspace`).

Snapshot `load` answers 409 while a task is running. A snapshot `save` waits for its context's running
task to end, and a conversation runs one task at a time.

Usage is the conversation's spend over the task: tokens, cost as LiteLLM prices it, and tool calls. A
failed run's error code is `openhands.<code>` of the SDK's last error event (`openhands.LLMAuthenticationError`,
`openhands.MaxIterationsReached`, ...). It is typed by the SDK's classification when the event carries one;
otherwise a run that raised is an `infra_error`, and one a run limit ended, or that ends stuck or paused, an
`agent_error`. A timeout (`openhands.timeout`) interrupts the conversation, which the context's next task resumes.

`tst/integration/cli/run_bundle_openhands_test.py` runs the built image against a fake OpenAI endpoint.

```bash
docker build -t openhands-a2a src/agent_env/agents/agents/openhands
docker run -p 8000:8000 -e LITELLM_BASE_URL=... -e LITELLM_API_KEY=... openhands-a2a
pytest tst/unit/agents/openhands
```

The OpenHands version is pinned in `requirements.txt` and `OPENHANDS_VERSION`; a new release is a pull
request that bumps both.
