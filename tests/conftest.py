"""Pytest configuration and global environment fixtures."""

import os

# Ensure isolated in-memory vector database during automated test runs
os.environ["QDRANT_HOST"] = ":memory:"
