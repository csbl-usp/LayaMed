# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use **Security → Report a vulnerability** on this repository (GitHub private
vulnerability reporting), with the steps to reproduce and the affected version or commit. We will answer as soon as
possible and agree on a fix and a disclosure date with you.

## What is in scope

- The local server (`server.py`): it is meant for this machine or a trusted private network only, without
  authentication. See [docs/08-run-locally.md](docs/08-run-locally.md).
- Text processing that a crafted input could abuse (for example, very slow regular expressions).
- Anything that could leak training data or model files through the API.

## Data

LayaMed must never receive real patient data in this repository. If you find restricted data (for example MIMIC
text or weights trained on it) in the repository or its history, report it the same way.
