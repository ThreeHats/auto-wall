# Auto-Wall Build Makefile

.PHONY: build install clean test help build-linux build-windows build-macos

# Default target
all: build

# Build for current platform
build:
	@echo "Building Auto-Wall for $(shell uname -s)..."
	python3 build.py

# Platform-specific builds
build-linux:
	python3 build.py --platform=linux

build-windows:
	python3 build.py --platform=windows

build-macos:
	python3 build.py --platform=macos

# Install locally (Linux/macOS only)
install: build
	@if [ -f "./install-linux.sh" ]; then \
		./install-linux.sh; \
	else \
		echo "No install script found for this platform"; \
	fi

# Install build dependencies
install-deps:
	python3 build.py --install-deps

# Development tasks
clean:
	python3 build.py --clean

test:
	python3 -m pytest tests/ -v

help:
	@echo "Auto-Wall Build System"
	@echo "====================="
	@echo "make build        - Build for current platform"
	@echo "make build-linux  - Build Linux packages"
	@echo "make build-windows - Build Windows executable"
	@echo "make build-macos  - Build macOS app bundle"
	@echo "make install      - Install locally after building"
	@echo "make install-deps - Install build dependencies"
	@echo "make clean        - Clean build artifacts"
	@echo "make test         - Run test suite"
	@echo ""
	@if [ -f auto_wall.py ]; then \
		echo "Current version: $$(grep 'APP_VERSION' auto_wall.py | cut -d'"' -f2)"; \
	fi