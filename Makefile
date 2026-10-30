```makefile
.PHONY: install inference analysis test clean

install:
	pip install -r requirements.txt

inference:
	bash scripts/run_inference.sh

analysis:
	python -m src.analysis --config configs/base.yaml

test:
	pytest tests/

clean:
	rm -rf results/*.csv results/*.png results/*.txt
