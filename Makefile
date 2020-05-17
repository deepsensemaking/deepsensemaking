TARGET="deepsensemaking"

help:
	@cat Makefile

all: clean build install

clean:
	@rm -fvR build
	@rm -fvR dist
	@rm -fvR __pycache__
	@rm -fvR src/${TARGET}.egg-info
	@find . -type f -name "*.~undo-tree~" -exec rm -vf {} \;

build:
	@python3 setup.py bdist_wheel

install:
	@pip install .

public:
	@python3 -m twine upload dist/*.whl
