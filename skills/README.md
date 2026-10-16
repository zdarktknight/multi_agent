# External Skills

Put customer or project-specific skill folders here.

The runtime loads bundled skills from `zhuge_agent/skills/` first, then recursively loads this external directory. You can also point to another extension directory with:

```bash
export ZHUGE_SKILLS_DIR=/path/to/custom/skills
```

External skill ids must not duplicate bundled skill ids.
