"""OpenHands (https://github.com/OpenHands/software-agent-sdk) as an AgentEnv A2A agent.

Each task is one run of an OpenHands SDK conversation in the workspace. A context maps to one persisted conversation,
so later tasks in the context resume it. This module is the A2A side and imports no SDK; ``openhands_runtime`` runs
the conversation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import shutil
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from agentenv_protocol import ROLE_HEADER
from agentenv_protocol.a2a_agent import (
    INSTALL_V1,
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    BundleSkillRequest,
    DataPart,
    FilePart,
    InlineSkillRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SKILL_CONFIG_V1,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    a2a_agent,
    download,
    enable,
    extension,
    upload,
)
from starlette.exceptions import HTTPException

OPENHANDS_VERSION = "1.57.0"
TRAJECTORY_FORMAT = "openhands-events/v1"
# The model when agent-config sets none, as for a peer that other agents message directly.
DEFAULT_MODEL_ENV = "OPENHANDS_A2A_MODEL"
A2A_PORT = 8000
INSTALL_DIR = "/opt/openhands-a2a"
ENV_FILE = "agent.env"
BASE_STATE = "base_state.json"
FETCH_TIMEOUT_SECONDS = 120
MAX_ATTACHMENT_BYTES = 256 * 1024 * 1024
# A conversation id is derived from its context id, so a context finds its conversation across restarts.
_CONVERSATIONS = uuid.uuid5(uuid.NAMESPACE_URL, "urn:agentenv:agents:openhands")
# Images go to the model as images as well as to disk; any other file is left on disk for the agent's tools.
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# sudo only where it needs no password: VM sandboxes have it, a local sandbox runs docker as the user.
_DOCKER = "$(sudo -n true 2>/dev/null && echo sudo) docker"
INSTALL_COMMANDS = [
    f"{_DOCKER} cp {{agent_ctx_tar}} {{container}}:/tmp/openhands-a2a-ctx.tar.gz",
    f"{_DOCKER} exec {{container}} sh -c 'mkdir -p {INSTALL_DIR} && tar -xf /tmp/openhands-a2a-ctx.tar.gz -C {INSTALL_DIR}"
    " && rm /tmp/openhands-a2a-ctx.tar.gz'",
    f"{_DOCKER} exec {{container}} sh {INSTALL_DIR}/install.sh",
    # The credentials travel on a heredoc into a 0600 file: the installer logs only a command's first line,
    # and neither a command line nor the container's config ever holds them.
    f"{_DOCKER} exec -i {{container}} sh -c 'umask 077 && cat > {INSTALL_DIR}/{ENV_FILE}' <<'OPENHANDS_A2A_ENV'\n"
    "LITELLM_API_KEY={litellm_api_key}\n"
    "LITELLM_BASE_URL={litellm_base_url}\n"
    "LITELLM_EXTRA_HEADERS={litellm_extra_headers}\n"
    "OPENHANDS_A2A_ENV",
    f"{_DOCKER} exec -d -e A2A_PORT={{a2a_port}} {{container}} sh {INSTALL_DIR}/start.sh",
]


class OpenHandsConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    max_iterations: int = 500
    timeout_seconds: int = 1800


@dataclass(frozen=True)
class Turn:
    """One task, as the runtime runs it."""

    conversation_id: str
    conversations_dir: Path
    workspace: Path
    text: str
    images: tuple[str, ...]
    mcp_servers: Mapping[str, Any]
    skills_dir: Path
    skill_names: tuple[str, ...]
    config: OpenHandsConfig
    base_url: str
    api_key: str | None
    extra_headers: Mapping[str, str]
    on_event: Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Outcome:
    text: str
    usage: Usage
    error_code: str | None = None
    error_type: Literal["agent_error", "infra_error"] = "agent_error"


class Runtime(Protocol):
    async def run(self, turn: Turn) -> Outcome:
        """Run ``turn`` to its end. Cancelled, it interrupts the conversation and re-raises once it has stopped."""
        ...


def conversation_id(context_id: str) -> str:
    return uuid.uuid5(_CONVERSATIONS, context_id).hex


def _with_role(mcp_servers: Mapping[str, Any], role: str | None) -> dict[str, Any]:
    """The servers with the role header, which the env gateway filters tools by and which wins over a stored role."""
    if not role:
        return dict(mcp_servers)
    return {
        name: {**registration, "headers": {**(registration.get("headers") or {}), ROLE_HEADER: role}}
        for name, registration in mcp_servers.items()
    }


def _media_type(part: FilePart) -> str:
    return part.mime_type or mimetypes.guess_type(part.name or "")[0] or "application/octet-stream"


async def _materialize(part: FilePart, index: int, directory: Path) -> Path:
    """Write a file part to disk; inline bytes and HTTP(S) URLs only."""
    name = Path(part.name).name if part.name else ""
    destination = directory / f"{index}-{name or 'file'}"
    if part.bytes is not None:
        destination.write_bytes(base64.b64decode(part.bytes))
        return destination
    if urlsplit(part.uri).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported file URI scheme for {name or 'file'}")
    received = 0
    async with (
        httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client,
        client.stream("GET", part.uri) as response,
    ):
        response.raise_for_status()
        with destination.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > MAX_ATTACHMENT_BYTES:
                    raise ValueError(f"{name or 'file'} is larger than {MAX_ATTACHMENT_BYTES} bytes")
                handle.write(chunk)
    return destination


def _pack(directory: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(directory.iterdir()):
            tar.add(child, arcname=child.name)


def _unpack(archive: Path, directory: Path) -> None:
    """Replace ``directory``'s contents with the archive's. The archive is extracted into a staging directory
    inside it first, so one that fails to extract leaves the contents as they were."""
    staging = directory / f".openhands-a2a-restore-{uuid.uuid4().hex}"
    try:
        # An empty workspace's archive has no members, and extracting it creates nothing.
        staging.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        for child in directory.iterdir():
            if child == staging:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in staging.iterdir():
            child.rename(directory / child.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _install_conversation(archive: Path, conversations_dir: Path, target_id: str) -> None:
    """Make the archived conversation the one ``target_id`` names: the SDK resumes a conversation only under the id
    and directory its state records."""
    target = conversations_dir / target_id
    staging = conversations_dir / f".{target_id}.{uuid.uuid4().hex}"
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        state = json.loads((staging / BASE_STATE).read_text())
        state.update(id=str(uuid.UUID(target_id)), persistence_dir=str(target))
        (staging / BASE_STATE).write_text(json.dumps(state))
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@a2a_agent(
    identity=AgentIdentity(
        name="OpenHands",
        description="The OpenHands agent: reads, edits and runs code in its workspace with MCP tools and skills.",
        version=OPENHANDS_VERSION,
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    config=OpenHandsConfig,
    config_description="Configure the OpenHands runtime: model, effort, system prompt and iteration limit.",
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True, description="The task's OpenHands SDK events, also while it runs."),
        enable(
            SNAPSHOT_V1,
            description=(
                "The trajectory object is the context's persisted OpenHands conversation, the workspace object a "
                "gzipped tar."
            ),
        ),
        PEER_AGENTS_V1,
        enable(
            INSTALL_V1,
            install_commands=INSTALL_COMMANDS,
            required_params=[
                "container", "agent_ctx_tar", "a2a_port", "litellm_api_key", "litellm_base_url",
                "litellm_extra_headers",
            ],
            a2a_port=A2A_PORT,
        ),
        TRIGGERS_V1,
    ),
)
class OpenHandsAgent(AgentEnvAgent):
    def __init__(self, runtime: Runtime, *, home: Path | None = None, workspace: Path | None = None) -> None:
        self.runtime = runtime
        self.home = home or Path.home() / ".openhands-a2a"
        self.workspace = workspace or Path("/workspace")
        self.conversations_dir = self.home / "conversations"
        self.skills_dir = self.home / "skills"
        self.inputs_dir = self.home / "inputs"
        # A conversation runs one task at a time; a snapshot of it waits for the task to end.
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        # Restoring the workspace waits for no task to be running in it; tasks and saves wait for a restore to finish.
        self._workspace_lock = asyncio.Lock()
        self._running = 0

    async def run(self, request: TaskRequest[OpenHandsConfig]) -> TaskResult:
        config = request.config
        if not config.model and (fallback := os.environ.get(DEFAULT_MODEL_ENV)):
            config = config.model_copy(update={"model": fallback})
        if not config.model:
            return TaskResult.failure("openhands.no_model", "no model is configured; set 'model' through agent-config")
        base_url = os.environ.get("LITELLM_BASE_URL")
        if not base_url:
            return TaskResult.failure("openhands.no_endpoint", "LITELLM_BASE_URL is not set", error_type="infra_error")
        workspace = request.workspace or self.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            text, images = await self._prompt(request)
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return TaskResult.failure("openhands.attachment", f"could not read a file part: {exc}")

        request.trajectory.set_format(TRAJECTORY_FORMAT)
        # The SDK calls back from its own threads too; the log takes one appender at a time.
        appending = threading.Lock()

        def on_event(event: dict[str, Any]) -> None:
            with appending:
                request.trajectory.append(event)

        conversation = conversation_id(request.context_id)
        turn = Turn(
            conversation_id=conversation,
            conversations_dir=self.conversations_dir,
            workspace=workspace,
            text=text,
            images=images,
            mcp_servers=_with_role(request.mcp_servers, config.role),
            skills_dir=self.skills_dir,
            skill_names=tuple(str(skill["name"]) for skill in request.skills),
            config=config,
            base_url=base_url,
            api_key=os.environ.get("LITELLM_API_KEY"),
            extra_headers=json.loads(os.environ.get("LITELLM_EXTRA_HEADERS") or "{}"),
            on_event=on_event,
        )
        async with self._workspace_lock:
            self._running += 1
        try:
            async with self._conversation_lock(conversation):
                try:
                    async with asyncio.timeout(config.timeout_seconds):
                        outcome = await self.runtime.run(turn)
                except TimeoutError:
                    message = f"OpenHands did not finish within {config.timeout_seconds}s"
                    return TaskResult.builder().failed("openhands.timeout", message).add_text(message).build()
        finally:
            self._running -= 1
        builder = TaskResult.builder().session_ref(conversation).usage(outcome.usage)
        if outcome.error_code is None:
            return builder.succeeded().add_text(outcome.text).build()
        return builder.failed(outcome.error_code, outcome.text, error_type=outcome.error_type).add_text(outcome.text).build()

    async def _prompt(self, request: TaskRequest[OpenHandsConfig]) -> tuple[str, tuple[str, ...]]:
        """The message text and its images as data URIs. Files are kept on disk for the conversation's later turns."""
        files_dir = self.inputs_dir / request.task_id
        chunks: list[str] = []
        images: list[str] = []
        for index, part in enumerate(request.parts):
            if isinstance(part, TextPart):
                chunks.append(part.text)
            elif isinstance(part, DataPart):
                chunks.append(json.dumps(part.data, indent=2, default=str))
            elif isinstance(part, FilePart):
                files_dir.mkdir(parents=True, exist_ok=True)
                path = await _materialize(part, index, files_dir)
                media_type = _media_type(part)
                if media_type in _IMAGE_TYPES:
                    images.append(f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}")
                chunks.append(f"[Attached file ({media_type}) saved at {path}]")
        return "\n\n".join(chunks), tuple(images)

    def _conversation_lock(self, conversation: str) -> asyncio.Lock:
        return self._conversation_locks.setdefault(conversation, asyncio.Lock())

    def _require_idle(self) -> None:
        if self._running:
            raise HTTPException(status_code=409, detail="a task is running; restore the workspace once it finishes")

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        conversation = conversation_id(request.context_id)
        directory = self.conversations_dir / conversation
        with tempfile.TemporaryDirectory() as scratch:
            # The conversation and the workspace are copied together, so a restore cannot fall between them.
            conversation_archive = Path(scratch) / "conversation.tar.gz"
            workspace_archive = Path(scratch) / "workspace.tar.gz"
            async with self._conversation_lock(conversation), self._workspace_lock:
                if not (directory / BASE_STATE).is_file():
                    raise HTTPException(status_code=404, detail=f"No conversation for context '{request.context_id}'")
                await asyncio.to_thread(_pack, directory, conversation_archive)
                if request.objects.workspace is not None:
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(_pack, self.workspace, workspace_archive)
            objects = {"trajectory": (await upload(request.objects.trajectory, conversation_archive)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, workspace_archive)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as scratch:
            conversation_archive = Path(scratch) / "conversation.tar.gz"
            await download(request.objects.trajectory, conversation_archive)
            if request.objects.workspace is not None:
                workspace_archive = Path(scratch) / "workspace.tar.gz"
                await download(request.objects.workspace, workspace_archive)
            async with self._workspace_lock:
                self._require_idle()
                if request.objects.workspace is not None:
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(_unpack, workspace_archive, self.workspace)
                self.conversations_dir.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    _install_conversation, conversation_archive, self.conversations_dir, conversation_id(context_id)
                )
        return {"context_id": context_id}

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, Any]:
        skill_dir = self.skills_dir / request.name
        skill_dir.mkdir(parents=True, exist_ok=False)
        (skill_dir / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, Any]:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        staging = self.skills_dir / f".{request.name}.{uuid.uuid4().hex}"
        try:
            for item in request.skill_bundle.files:
                destination = (staging / item.path).resolve()
                if not destination.is_relative_to(staging.resolve()):
                    raise ValueError(f"skill file {item.path} escapes the skill directory")
                await download(item.object, destination)
            staging.rename(self.skills_dir / request.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"name": request.name}
