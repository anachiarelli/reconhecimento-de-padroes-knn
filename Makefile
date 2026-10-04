run:
	docker run --rm -w /usr/src/app -v .:/usr/src/app python:3.14 python main.py

install:
	docker run --rm -u 1000 -w /usr/src/app -v .:/usr/src/app python:3.14 python3 -m venv .venv
	docker run --rm -u 1000 -w /usr/src/app -v .:/usr/src/app python:3.14 bash -c "source .venv/bin/activate && pip3 install -U scikit-learn"
