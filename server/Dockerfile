FROM dailyco/pipecat-base:latest

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy all application code
COPY ./bot.py bot.py
COPY ./mock_backend.py mock_backend.py
COPY ./police_agent.py police_agent.py
COPY ./report_generator.py report_generator.py
COPY ./nemotron_llm.py nemotron_llm.py
COPY ./nvidia_stt.py nvidia_stt.py
COPY ./cekura_eval.py cekura_eval.py

# Copy directories
COPY ./templates templates
COPY ./mock_data mock_data
COPY ./improvement improvement
COPY ./review_ui review_ui
