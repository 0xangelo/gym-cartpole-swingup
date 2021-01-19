bump-patch:
	poetry version patch
	git add pyproject.toml
	git commit -s -m "chore: bump version patch"

bump-minor:
	poetry version minor
	git add pyproject.toml
	git commit -s -m "chore: bump version minor"

bump-major:
	poetry version major
	git add pyproject.toml
	git commit -s -m "chore: bump version major"

push-release:
	git push origin master develop --tags

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
