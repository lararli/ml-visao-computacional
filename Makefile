# CI/CD simulado (PADRAO_DESENVOLVIMENTO.md). Rode `make help` para a lista.
# O venv fica em .venv.nosync: o iCloud não sincroniza pastas .nosync.
SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
export UV_PROJECT_ENVIRONMENT := .venv.nosync
export PYTHONDONTWRITEBYTECODE := 1
RUN := $(UV) run --frozen

.PHONY: help setup lint format test smoke ci data pipeline gold run run-a1-scratch \
        run-a1-pretrained run-a2 run-a3 run-a4 all clean

help:
	@echo "setup lint test smoke ci data pipeline gold all clean"
	@echo "um dataset só: make pipeline ONLY=austin_houses (ou ads16, images, chest_xray); treinar de novo: FORCE=1"

setup:  ## cria .venv.nosync e instala as dependências do uv.lock
	$(UV) sync --frozen
	@# .venv aponta para .venv.nosync: um `uv run` sem a variável não cria venv dentro do iCloud
	@ln -sfn .venv.nosync .venv

lint:  ## ruff check e ruff format --check
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format:  ## aplica o ruff format
	$(RUN) ruff format src tests scripts

test:  ## testes unitários: sem rede, sem GPU, sem dados baixados
	$(RUN) pytest -q -m "not smoke and not data"

smoke:  ## pipeline inteiro com uma amostra minúscula, em CPU
	$(RUN) pytest -q -m smoke

ci: lint test smoke  ## lint + test + smoke

data:  ## baixa, verifica e grava o manifesto (idempotente); A1: consultas, catálogo e recortes
	$(RUN) python scripts/download_data.py
	$(RUN) python scripts/build_a1.py

ONLY ?=
FORCE ?=
pipeline:  ## por dataset: silver -> treino -> avaliação -> gold (treina só o que falta em outputs/)
	$(RUN) python scripts/run_pipeline.py $(if $(ONLY),--only $(ONLY),) $(if $(FORCE),--force,)

gold:  ## só publica a gold dos 4 datasets a partir de outputs/ (não treina)
	$(RUN) python scripts/run_pipeline.py --gold-only

run: pipeline  ## nome antigo do pipeline

run-a1-scratch:  ## A1: ViT do zero (piscina e solar), métricas, mapas de atenção e reports/sections/a1.md
	$(RUN) python scripts/run_a1_scratch.py

run-a1-pretrained:  ## A1: ViT pré-treinado e ResNet-18, comparação, "não dá" com motivo e gold
	$(RUN) python scripts/run_a1_pretrained.py

run-a2:  ## CLIP no ADS-16: embeddings no DuckDB, ranking de objetos, busca por texto
	$(RUN) python scripts/run_a2.py

run-a3:  ## A3: CNN pré-treinada congelada (treino, métricas, curvas e a seção do relatório)
	$(RUN) python scripts/run_a3.py

run-a4:  ## A4.1: caso de raio-X, GAN sem e com mitigação, recall com e sem sintético, a4.md
	$(RUN) python scripts/run_a4.py

all: setup data pipeline  ## do download à gold: dados, treinos, avaliação e tabelas gold

clean:  ## apaga o que é gerado; nunca data.nosync/raw/
	rm -rf outputs .pytest_cache .ruff_cache
	rm -rf $${VISAOCOMP_DATA_DIR:-data.nosync}/interim $${VISAOCOMP_DATA_DIR:-data.nosync}/processed
	find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
