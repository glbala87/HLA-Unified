.PHONY: test test-unit test-integration lint validate validate-real docker clean install dev

# ── Install ──────────────────────────────────────────────────────────
install:
	pip install -e .

dev:
	pip install -e ".[dev,benchmark]"

# ── Test ─────────────────────────────────────────────────────────────
test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=hla_unified --cov-report=term-missing

# ── Lint ─────────────────────────────────────────────────────────────
lint:
	python -c "import hla_unified; print(f'v{hla_unified.__version__}')"
	python -c "from hla_unified.pipeline.runner import UnifiedPipeline"
	python -c "from hla_unified.benchmark.runner import BenchmarkRunner"

# ── Validation ───────────────────────────────────────────────────────
validate:
	python validation/run_validation.py

validate-real:
	python validation/real_data/run_real_validation.py \
		--samples NA12878 \
		--imgt-db IMGTHLA \
		--out validation/real_data/results \
		--threads 4 --region-only

validate-real-full:
	python validation/real_data/run_real_validation.py \
		--samples NA12878,HG002,HG003,HG004,NA19240 \
		--imgt-db IMGTHLA \
		--out validation/real_data/results \
		--threads 8

# ── Docker ───────────────────────────────────────────────────────────
docker:
	docker build -f Dockerfile.unified -t hla-unified:2.0.0 .

docker-test: docker
	docker run --rm hla-unified:2.0.0 --help

# ── Database ─────────────────────────────────────────────────────────
setup-db:
	python -m hla_unified setup-db --release 3.56.0

# ── Clean ────────────────────────────────────────────────────────────
clean:
	rm -rf __pycache__ .pytest_cache hla_unified/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
