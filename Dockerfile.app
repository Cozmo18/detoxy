# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install curl
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra cpu --locked --no-install-project --no-dev 

# Copy only the required files and directories
COPY detoxy/app/ /app/detoxy/app/
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY .python-version /app/.python-version

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra cpu --locked --no-dev


# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

EXPOSE 8000
CMD ["python", "/app/detoxy/app/main.py"]