# EDA Multi-Agent System

An intelligent Exploratory Data Analysis system powered by multiple AI agents, built with LangGraph, A2A Protocol, MCP Server, and AG-UI.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          AG-UI                                  │
│                    (Streaming Interface)                        │
├─────────────────────────────────────────────────────────────────┤
│                        LangGraph                                │
│                     (Orchestration)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Planner  │←→│  Critic  │←→│  Code    │←→│ Analyst  │       │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                         ↕ A2A Protocol                         │
├─────────────────────────────────────────────────────────────────┤
│                       MCP Server                                │
│           (SQL Execution, Python Sandbox, Tools)                │
├─────────────────────────────────────────────────────────────────┤
│      Neo4j (GraphRAG)    │    PostgreSQL (Metadata)            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL
- Neo4j
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "EDA agent"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt

# Copy environment file
copy .env.example .env
# Edit .env with your credentials
```

### Configuration

Edit `.env` file with your settings:

```env
OPENAI_API_KEY=your-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password
POSTGRES_PASSWORD=your-password
```

### Running

```bash
# Start the API server
python -m src.api.main
# or
eda-agent

# Server will be available at http://localhost:8000
```

## 📁 Project Structure

```
src/
├── agents/          # AI Agents (Planner, Critic, Code, Analyst)
├── graph/           # LangGraph workflow orchestration
├── protocols/       # A2A and AG-UI protocol implementations
├── mcp/             # MCP Server and tools
├── memory/          # GraphRAG, Episodic Memory, Metadata Store
├── models/          # Data models
├── api/             # FastAPI application
└── utils/           # Utilities
```

## 🤖 Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Planner** | Data Scientist | Generates hypotheses and analysis plans |
| **Critic** | Business Expert | Validates plans against schema and rules |
| **Code Agent** | Developer | Generates and executes SQL/Python code |
| **Analyst** | Data Analyst | Evaluates results and generates insights |

## 📡 API Endpoints

- `POST /api/v1/analyze` - Analyze a question (streaming SSE)
- `GET /api/v1/health` - Health check
- `GET /api/v1/sessions/{id}` - Get session details

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Agents Guide](docs/agents.md)
- [API Reference](docs/api.md)

## 📄 License

MIT License
