.PHONY: install prepare train-clinical train-image explain serve app test clean

install:
	python -m pip install -r requirements.txt

prepare:
	python -m scripts.prepare_clinical

train-clinical:
	python -m scripts.train_clinical

train-image:
	python -m scripts.train_image

explain:
	python -m scripts.explain_all

serve:
	python -m uvicorn src.api.main:app --reload --port 8000

app:
	python -m streamlit run app/streamlit_app.py

test:
	python -m pytest

clean:
	python -c "import pathlib, shutil; root = pathlib.Path('.'); dirs = [p for p in root.rglob('__pycache__')]; dirs.append(root / '.pytest_cache'); [shutil.rmtree(p, ignore_errors=True) for p in dirs]; art = root / 'artifacts'; [p.unlink() for p in art.rglob('*') if p.is_file() and p.name != '.gitkeep']"
