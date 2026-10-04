.PHONY: doctor test bench serve train install

install:
	pip install -e .

doctor:
	python -m kolibriforge doctor

test:
	python -m unittest discover -s tests -t .

bench:
	python -m kolibriforge evaluate

train:
	python -m kolibriforge train

serve:
	python -m kolibriforge serve --host 0.0.0.0 --port 8000
