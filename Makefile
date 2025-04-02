# Makefile for intura-ai package

.PHONY: help clean build test lint deploy bump-patch bump-minor bump-major release-patch release-minor release-major

PACKAGE_NAME := intura-ai
VERSION_FILE := intura_ai/__version__.py

help:
	@echo "intura-ai package management commands:"
	@echo ""
	@echo "  clean        - Remove build artifacts and cache directories"
	@echo "  lint         - Run linting checks"
	@echo "  test         - Run tests"
	@echo "  build        - Build package distribution files"
	@echo "  deploy       - Deploy to PyPI (requires PyPI credentials)"
	@echo ""
	@echo "Version commands:"
	@echo "  bump-patch   - Bump patch version (0.0.x)"
	@echo "  bump-minor   - Bump minor version (0.x.0)"
	@echo "  bump-major   - Bump major version (x.0.0)"
	@echo ""
	@echo "Release commands (version + build + deploy + tag):"
	@echo "  release-patch - Release a patch version"
	@echo "  release-minor - Release a minor version"
	@echo "  release-major - Release a major version"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@echo "Running linters..."
	python -m black intura_ai tests
	python -m isort intura_ai tests
	python -m flake8 intura_ai tests
	python -m mypy intura_ai

test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=intura_ai

build: clean
	@echo "Building distribution files..."
	python -m build

deploy: build
	@echo "Deploying to PyPI..."
	python -m twine upload dist/*

# Version management functions
define get_version
	$(shell grep -o '".*"' $(VERSION_FILE) | tr -d '"')
endef

define update_version
	@echo 'Updating version to $(1)'
	@echo '__version__ = "$(1)"' > $(VERSION_FILE)
	@git add $(VERSION_FILE)
	@git commit -m "Bump version to $(1)"
endef

bump-patch:
	$(eval CURRENT_VERSION := $(call get_version))
	$(eval MAJOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f1))
	$(eval MINOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f2))
	$(eval PATCH := $(shell echo $(CURRENT_VERSION) | cut -d. -f3))
	$(eval NEW_PATCH := $(shell echo $$(($(PATCH) + 1))))
	$(eval NEW_VERSION := $(MAJOR).$(MINOR).$(NEW_PATCH))
	$(call update_version,$(NEW_VERSION))
	@echo "Version bumped to $(NEW_VERSION)"

bump-minor:
	$(eval CURRENT_VERSION := $(call get_version))
	$(eval MAJOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f1))
	$(eval MINOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f2))
	$(eval NEW_MINOR := $(shell echo $$(($(MINOR) + 1))))
	$(eval NEW_VERSION := $(MAJOR).$(NEW_MINOR).0)
	$(call update_version,$(NEW_VERSION))
	@echo "Version bumped to $(NEW_VERSION)"

bump-major:
	$(eval CURRENT_VERSION := $(call get_version))
	$(eval MAJOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f1))
	$(eval NEW_MAJOR := $(shell echo $$(($(MAJOR) + 1))))
	$(eval NEW_VERSION := $(NEW_MAJOR).0.0)
	$(call update_version,$(NEW_VERSION))
	@echo "Version bumped to $(NEW_VERSION)"

create-tag:
	$(eval CURRENT_VERSION := $(call get_version))
	@echo "Creating tag v$(CURRENT_VERSION)"
	@git tag -a "v$(CURRENT_VERSION)" -m "Version $(CURRENT_VERSION)"
	@git push origin "v$(CURRENT_VERSION)"

release-patch: bump-patch build deploy create-tag

release-minor: bump-minor build deploy create-tag

release-major: bump-major build deploy create-tag