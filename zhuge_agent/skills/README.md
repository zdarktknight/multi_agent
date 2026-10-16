# Bundled Skills

This directory contains built-in Zhuge Agent skills shipped with the Python package.

Each skill is an Anthropic-style folder:

- `SKILL.md`: LLM-facing skill description and operating instructions.
- `skill.json`: Runtime-facing execution graph, tool bindings, and output type.

The runtime loads this directory recursively before loading external extension skills.
