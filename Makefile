.PHONY: setup data train test lint api dashboard up

setup:
	python -m pip install -e '.[dev]'

data:
	PYTHONPATH=src python scripts/generate_demo_data.py

train: data
	PYTHONPATH=src python scripts/train_model.py

test: train
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests scripts

api: train
	PYTHONPATH=src uvicorn incidentsense.api:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

up:
	docker compose up --build
