"""The OpenHands SDK side of the agent: a task is one ``arun()`` of the context's persisted conversation, with the
default tools, the task's MCP servers and its skills."""

import asyncio
import uuid
from typing import Any, Final

from agentenv_protocol.a2a_agent import Usage
from openhands.sdk import LLM, Agent, AgentContext, Conversation, ImageContent, Message, TextContent
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, Event
from openhands.sdk.event.error_classification import FailureKind
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.mcp import MCPServer
from openhands.sdk.skills.skill import load_skills_from_dir
from openhands.tools.preset.default import get_default_tools

from openhands_agent import Outcome, Turn

# The AgentEnv endpoint is a LiteLLM gateway; this provider prefix sends LiteLLM's requests to it as is.
_PROVIDER: Final = "litellm_proxy"


def _llm(turn: Turn) -> LLM:
    options: dict[str, Any] = {}
    if turn.config.effort is not None:
        options["reasoning_effort"] = turn.config.effort
    return LLM(
        usage_id="agent",
        model=f"{_PROVIDER}/{turn.config.model}",
        base_url=turn.base_url,
        api_key=turn.api_key,
        extra_headers=dict(turn.extra_headers) or None,
        **options,
    )


def _agent(turn: Turn) -> Agent:
    skills = []
    if turn.skill_names and turn.skills_dir.is_dir():
        loaded = load_skills_from_dir(turn.skills_dir, strict=False)
        skills = [skill for group in loaded for skill in group.values() if skill.name in turn.skill_names]
    mcp_config = {
        name: MCPServer(url=registration["url"], transport="http", headers=registration.get("headers") or None)
        for name, registration in turn.mcp_servers.items()
    }
    options: dict[str, Any] = {}
    if turn.config.system_prompt:
        options["system_prompt"] = turn.config.system_prompt
    return Agent(
        llm=_llm(turn),
        tools=get_default_tools(enable_browser=False),
        mcp_config=mcp_config,
        agent_context=AgentContext(
            skills=skills,
            system_message_suffix=turn.config.append_system_prompt,
            load_user_skills=False,
            load_public_skills=False,
        ),
        **options,
    )


def _usage(before: Metrics, after: Metrics, tool_calls: int) -> Usage:
    def spent(name: str) -> int:
        return (getattr(after.accumulated_token_usage, name, 0) or 0) - (
            getattr(before.accumulated_token_usage, name, 0) or 0
        )

    input_tokens, output_tokens = spent("prompt_tokens"), spent("completion_tokens")
    cost = after.accumulated_cost - before.accumulated_cost
    return Usage(
        tool_call_count=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost or None,
        provider_details={
            "cache_read_tokens": spent("cache_read_tokens"),
            "cache_write_tokens": spent("cache_write_tokens"),
            "reasoning_tokens": spent("reasoning_tokens"),
        },
    )


async def _run_to_its_end(conversation: Any) -> None:
    """``arun()`` on an event loop of its own in a worker thread, interrupted when the caller is cancelled and awaited
    until it has stopped. The SDK does blocking work between its awaits (tools, persistence), which on the server's
    loop would stall every other task and the agent's own HTTP endpoints, the peer-agents MCP server among them."""
    runner = asyncio.Runner()
    loop = runner.get_loop()

    def drive() -> None:
        # Closing the runner cancels what the run left on its loop, LiteLLM's logging worker among them.
        with runner:
            runner.run(conversation.arun())

    running = asyncio.ensure_future(asyncio.to_thread(drive))
    try:
        await asyncio.shield(running)
    except asyncio.CancelledError:
        if not running.done():
            loop.call_soon_threadsafe(conversation.interrupt)
            await asyncio.wait([running])
        raise


class SdkRuntime:
    async def run(self, turn: Turn) -> Outcome:
        actions = 0

        def on_event(event: Event) -> None:
            nonlocal actions
            if isinstance(event, ActionEvent):
                actions += 1
            turn.on_event(event.model_dump(mode="json"))

        def open_conversation() -> Any:
            return Conversation(
                agent=_agent(turn),
                workspace=str(turn.workspace),
                persistence_dir=str(turn.conversations_dir),
                conversation_id=uuid.UUID(turn.conversation_id),
                callbacks=[on_event],
                max_iteration_per_run=turn.config.max_iterations,
                visualizer=None,
                delete_on_close=False,
            )

        conversation = await asyncio.to_thread(open_conversation)
        try:
            before = conversation.conversation_stats.get_combined_metrics().model_copy(deep=True)
            content = [TextContent(text=turn.text), *(ImageContent(image_urls=[image]) for image in turn.images)]
            await asyncio.to_thread(conversation.send_message, Message(role="user", content=content))
            try:
                await _run_to_its_end(conversation)
            except ConversationRunError as exc:
                usage = _usage(before, conversation.conversation_stats.get_combined_metrics(), actions)
                error = exc.conversation_error
                if error is None:
                    return Outcome(f"OpenHands failed: {exc.original_exception}", usage, "openhands.error", "infra_error")
                kind = error.classification.kind if error.classification else None
                error_type = "agent_error" if kind == FailureKind.AGENT_ACTION else "infra_error"
                return Outcome(f"OpenHands failed: {error.detail}", usage, f"openhands.{error.code}", error_type)
            usage = _usage(before, conversation.conversation_stats.get_combined_metrics(), actions)
            status = conversation.state.execution_status
            if status != ConversationExecutionStatus.FINISHED:
                return Outcome(f"OpenHands stopped with status '{status.value}'", usage, f"openhands.{status.value}")
            reply = get_agent_final_response(conversation.state.events)
            return Outcome(reply or "The agent finished without a reply.", usage)
        finally:
            await asyncio.to_thread(conversation.close)
