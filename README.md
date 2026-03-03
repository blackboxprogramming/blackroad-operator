# BlackRoad Operator

Self-hosted operator service running on the BlackRoad Pi cluster. Zero external AI providers — all inference, automation, and orchestration runs on your own hardware.

## Architecture

```
Pi Cluster (LAN 192.168.4.x)
├── alice   (192.168.4.49) — primary LLM host (Ollama)
├── aria    (192.168.4.38) — agent fleet
└── lucidia (192.168.4.99) — auxiliary
```

## API Endpoints

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/health`         | Operator health check              |
| GET    | `/llm/health`     | Pi cluster LLM readiness           |
| GET    | `/cluster/health` | All Pi node connectivity status    |
| POST   | `/tasks`          | Execute browser automation task    |
| POST   | `/tools/sim.run`  | Gymnasium environment simulation   |
| POST   | `/tools/code.run` | Docker-isolated Python execution   |
| POST   | `/tools/git.pr`   | Create git branches and PRs        |

## Environment Variables

| Variable          | Default                | Description                     |
|-------------------|------------------------|---------------------------------|
| `OP_DB_DSN`       | `postgresql:///operator` | PostgreSQL connection string  |
| `OP_PI_LLM_HOST`  | `192.168.4.49`         | Pi node running Ollama          |
| `OP_PI_LLM_PORT`  | `11434`                | Ollama port on Pi node          |
| `OP_LLM_MODEL`    | `qwen2.5:0.5b`        | Model loaded in Ollama          |

## Running

```bash
uvicorn op_server:app --host 0.0.0.0 --port 8000
```

## License

BlackRoad OS, Inc. Proprietary — see [LICENSE](LICENSE).
