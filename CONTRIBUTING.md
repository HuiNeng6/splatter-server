# Contributing to splatter-server

Thank you for your interest in contributing. We welcome issues and pull requests.

## How to contribute

- **Bug reports and feature ideas:** Open an [issue](https://github.com/aukilabs/splatter-server/issues).
- **Code changes:** Open a pull request (PR) against `main`. Use a branch name that matches our CI (e.g. `feature/your-feature`, `bug/fix-description`, `chore/your-change`).

## Development setup

- **Rust (compute node):** See [server/rust/README.md](server/rust/README.md). Copy `server/rust/.env.example` to `server/rust/.env` and configure for local DDS/DMS if needed.
- **Docker:** From the repo root, `docker build -t splatter-server .` then run with `--env-file .env` as in the main [README](README.md).

## Code standards

- **Rust:** In `server/rust`, please run before pushing:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- CI runs the same checks on push; keep the pipeline green.

## Pull request process

1. Point your PR at the `main` branch.
2. Ensure CI passes (Rust format, clippy, tests).
3. Keep changes focused; link related issues where applicable.

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers this project.
