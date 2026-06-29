
build: make_editable build_wheel twine_check

bump_version_major:
	bumpver update --major

bump_version_minor:
	bumpver update --minor

bump_version_patch:
	bumpver update --patch

make_editable:
	python3 -m pip install -e .

build_wheel:
	poetry lock
	poetry build

twine_check:
	twine check dist/*

# Credentials are read by twine from the environment or from ~/.pypirc --
# never hardcode a token here. To publish:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=<your PyPI/TestPyPI API token>
# or configure ~/.pypirc with [pypi] / [testpypi] sections instead.
twine_test:
	twine upload -r testpypi dist/* --verbose

twine_upload:
	twine upload -r pypi dist/* --verbose
