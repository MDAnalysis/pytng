init:
	pip install -r requirements.txt

test:
	pytest tests

.PHONY: init test
