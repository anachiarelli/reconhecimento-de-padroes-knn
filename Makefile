split:
	docker run --rm -w /usr/src/app -v .:/usr/src/app python:3.14 bash -c "source .venv/bin/activate && python main.py 0.25 0.25 0.5"

lazy-learning:
	docker run --rm -w /usr/src/app -v .:/usr/src/app python:3.14 bash -c "source .venv/bin/activate && python lazy_learning.py"

alternative:
	docker run --rm -w /usr/src/app -v .:/usr/src/app python:3.14 bash -c "source .venv/bin/activate && python alternative.py 5"

install:
	docker run --rm -u 1000 -w /usr/src/app -v .:/usr/src/app python:3.14 python3 -m venv .venv
	docker run --rm -u 1000 -w /usr/src/app -v .:/usr/src/app python:3.14 bash -c "source .venv/bin/activate && pip3 install -U scikit-learn"
