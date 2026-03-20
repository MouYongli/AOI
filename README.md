# AIO — AI On-premise

Local-first AI Agent Gateway. Empowering everyone with a private, infinitely extensible AI assistant running on their own device.

## Features

- **Planner/Orchestrator Agent Architecture** — Root Agent with sub-agent spawning (template + dynamic), independent ReAct loops
- **Multi-Channel** — Telegram Bot (multi-session + real-time progress) and CLI (`aio chat` / `aio ask`)
- **MCP Tool Integration** — File system, terminal, web search, browser automation
- **Progressive Disclosure (SKILL)** — Context-aware dynamic tool set adjustment
- **Autonomy Levels** — `full-auto` / `semi-auto` / `user-confirm`, three-tier configurable
- **Multi-LLM Backend** — Ollama, OpenAI, Claude, Azure OpenAI
- **One-Click Deploy** — `docker-compose up` to start everything

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/MouYongli/AOI.git
cd AOI

# 2. Copy environment config
cp config/.env.example .env

# 3. Edit .env with your API keys
#    (At minimum, configure one LLM backend)

# 4. Start with Docker Compose
cd docker && docker-compose up -d

# 5. Use the CLI
aio chat
```

## Project Structure

```
aio/                    # Python main package
├── core/               # Agent core (orchestrator, agent_instance, skill, context, autonomy, message_bus)
├── channels/           # Channel adapters (telegram, cli, progress_reporter)
├── workflow/           # Workflow execution (sequential V1, DAG V2)
├── execution/          # Execution environments (host, sandbox, security_guard)
├── mcp/                # MCP integration (client, servers/)
├── llm/                # LLM backends (ollama, openai, claude, azure, registry)
├── models/             # SQLAlchemy ORM models
├── db/                 # Database engine, migrations
├── config/             # Configuration (Pydantic + YAML)
├── app.py              # FastAPI application
└── cli.py              # Typer CLI entry point
config/                 # YAML configuration files
docker/                 # Dockerfile + docker-compose.yaml
tests/                  # Test suite
docs/                   # Documentation
```

## Configuration

AIO uses YAML configuration (`config/config.yaml`) with Pydantic validation. See the config file for all available options.

Key configuration areas:
- LLM providers (Ollama, OpenAI, Claude, Azure OpenAI)
- Telegram Bot settings
- MCP server connections
- Security rules (allowed directories, blocked commands)
- Agent templates
- Autonomy levels

## Roadmap

| Phase | Focus |
|-------|-------|
| **Phase 1 (MVP)** | Core gateway: Planner/Orchestrator, Telegram + CLI, MCP tools, SKILL engine, Autonomy levels |
| **Phase 2** | Web UI, DAG workflow engine, RAG knowledge service |
| **Phase 3** | Plugin SDK, multi-tenant + RBAC, Slack Bot, one-click installer |

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy 2.0, asyncio
- **Database**: PostgreSQL 16
- **LLM**: Ollama, OpenAI, Claude, Azure OpenAI
- **MCP**: mcp-python SDK
- **Telegram**: python-telegram-bot
- **CLI**: Typer + Rich
- **Logging**: structlog
- **Testing**: pytest + pytest-asyncio
- **Deploy**: Docker Compose

## License

Apache-2.0
