# Development rules

- Read git status and this file before work; protect unrelated changes.
- Python is 3.12.13, managed only with uv and the committed lockfile.
- Use TDD for behavioral changes: intended Red before implementation, Green and regression. Record evidence in Japanese docs. Additional verification of existing code is not TDD.
- Keep README.md in English, README.ja.md in Japanese. Separate synthetic/host/target/device/human evidence.
- Never publish credentials, raw device data, recordings or private Lab history. Training data must have documented licenses and provenance. Model/audio licensing is separate from code.
- This project's Vast.ai budget is cumulative USD 100 across voice, behavior, retries and OS integration. No split repository budget reset. No paid job before tested quotation/reservation/deadline/cleanup guards. CPU-only training has no Vast.ai charge.
- Device firmware changes, PSRAM enablement, unknown GPIO, security/eFuse and raw recording transfer require concrete individual review and permission. Do not expand this project's authorization to those operations.
- Shared OS/SDK changes belong in scratch-arcade-os through a PR, required CI and normal merge.
- Inspect task diffs, run proportionate checks, commit and push coherent changes. No history rewriting or forced pushes. Keep main protected after CI is established.
