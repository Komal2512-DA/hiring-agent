from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from app.config import get_settings
from app.env import HiringOpenEnv
from app.models import Action, EnvironmentState, ResetRequest, ResetResponse, StepResponse

settings = get_settings()
env = HiringOpenEnv(seed=settings.openenv_seed)

app = FastAPI(title="Hiring Agent OpenEnv", version="1.0.0")


@app.get("/")
def root() -> dict:
    return {
        "service": "hiring-agent-openenv",
        "status": "ok",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/spaces/{owner}/{repo}")
@app.get("/spaces/{owner}/{repo}/")
def hf_spaces_compat(owner: str, repo: str) -> dict:
    # Compatibility for URL paths that may be forwarded by some HF wrappers.
    return root()


@app.get("/web")
@app.get("/web/")
@app.get("/web/{path:path}")
def hf_web_compat(path: str = "") -> dict:
    # OpenEnv push may set `base_path: /web` in HF README frontmatter.
    # Serve a valid payload for that path to avoid 404 in the Space app pane.
    return root()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> list[dict]:
    return [task.model_dump(mode="json") for task in env.list_tasks()]


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest | None = Body(default=None)) -> ResetResponse:
    task_id = request.task_id if request else None
    try:
        observation = env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation, state=env.state())


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(observation=observation, reward=reward)


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    return env.state()
