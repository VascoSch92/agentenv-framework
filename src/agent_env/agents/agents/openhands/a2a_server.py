"""Serves the OpenHands agent. ``AGENT_CARD`` is what agent-env's install/v1 reads from ``/app/a2a_server.py``."""

import os
from pathlib import Path

from openhands_agent import OpenHandsAgent
from openhands_runtime import SdkRuntime

agent = OpenHandsAgent(SdkRuntime(), workspace=Path(os.environ.get("OPENHANDS_A2A_WORKSPACE", "/workspace")))
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
