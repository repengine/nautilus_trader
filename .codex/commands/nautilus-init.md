---
description: Initialize Nautilus Codex session by loading required Nautilus skills
---

Initialize this session for Nautilus ML work.

1) Verify Nautilus skills exist under `~/.codex/skills/`:

- `nautilus-domain-patterns`
- `nautilus-project-navigator`
- `nautilus-test-writer`
- `nautilus-store-schemas`

2) Load each skill using the superpowers helper:

- `~/.codex/superpowers/.codex/superpowers-codex use-skill nautilus-domain-patterns`
- `~/.codex/superpowers/.codex/superpowers-codex use-skill nautilus-project-navigator`
- `~/.codex/superpowers/.codex/superpowers-codex use-skill nautilus-test-writer`
- `~/.codex/superpowers/.codex/superpowers-codex use-skill nautilus-store-schemas`

3) After loading, announce: "Nautilus skills loaded; following Nautilus ML rules for this session."

4) Proceed with user request, using the loaded Nautilus skills alongside superpowers process skills.
