.SHELL=/bin/bash

release:
	poetry run python distortion/distortion.py
	poetry version patch
	poetry publish --build -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
