# Holmes 0.46.0 maintenance branch

`codex/mimi-0.46` follows the official `0.46.0` release. Each runtime change is a separate commit so it can be compared with the next upstream release. The MCP optional-null fix is backported from [upstream #2695](https://github.com/HolmesGPT/holmesgpt/pull/2695).

## Build and verify

```sh
docker build -f Dockerfile.mimi --build-arg VCS_REF="$(git rev-parse HEAD)" -t holmesgpt:0.46.0-mimi.1 .
docker run --rm --network none \
  -e LITELLM_LOCAL_MODEL_COST_MAP=True -e LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True \
  -v "$PWD/tests/mimi:/suite:ro" --entrypoint python holmesgpt:0.46.0-mimi.1 \
  -B -m unittest discover -s /suite -v
```

The build uses the upstream image pinned by digest for dependencies and release metadata. It copies the agent source into the image at build time. Deployment settings and credentials are supplied by the operator.

`HOLMES_MAX_INPUT_TOKENS` sets a cap on input after reserving the model's output allowance. Zero uses the upstream model-window compaction rule. `HOLMES_ISOLATE_EMPTY_FRONTEND_TOOLS=true` gives requests with an empty frontend-tool list an empty executor. MCP toolsets may set `call_timeout_seconds` to limit the time spent opening a session and running one request.
