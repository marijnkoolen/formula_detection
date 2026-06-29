
build: make_editable build_wheel check

bump_version_major:
	bumpver update --major --no-push

bump_version_minor:
	bumpver update --minor --no-push

bump_version_patch:
	bumpver update --patch --no-push

make_editable:
	python3 -m pip install -e .

build_wheel:
	poetry lock
	poetry build

check:
	twine check dist/*

# Credentials are configured once via `poetry config pypi-token.pypi <token>`
# (and `poetry config pypi-token.testpypi <token>` for the test repository,
# after `poetry config repositories.testpypi https://test.pypi.org/legacy/`)
# -- never hardcode a token in this file or commit one anywhere in the repo.
publish_testpypi:
	poetry publish -r testpypi

publish_pypi:
	poetry publish
