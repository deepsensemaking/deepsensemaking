TARGET="deepsensemaking"

help:
	@cat Makefile

all: clean build install import-test

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

import-test:
	python -c "import deepsensemaking as dsm; print(dsm.__version__)"


public: all
	@python3 -m twine upload dist/*.whl

## on buka2 use:
## pip install git+ssh://jiko@stardust/home/jiko/cc/dev/py/major/2020-05-17__deepsensemaking/dsm_repo
## PROBLEM - no updates even with "--no-cache-dir" option

rsync-to-buka2:
	rsync \
	  --recursive \
	  --links \
	  --perms \
	  --times \
	  --group \
	  --owner \
	  --devices \
	  --specials \
	  --itemize-changes \
	  --human-readable \
	  --stats \
	  --progress \
	  --rsh=ssh \
	  "/home/jiko/cc/dev/py/major/2020-05-17__deepsensemaking/dsm_repo/" \
	  "eben@buka2:/home/eben/dsm_repo/"
