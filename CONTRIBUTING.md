# Contributing to νgrεp

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Rust 1.75+
- LLVM/Clang
- CMake

### Building

```bash
git clone https://github.com/CortexLM/vgrep.git
cd vgrep

# Setup git hooks
./scripts/setup-hooks.sh  # Unix
./scripts/setup-hooks.ps1 # Windows

cargo build
cargo test
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Write tests for new functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run lints (`cargo fmt && cargo clippy`)
6. Commit (`git commit -m 'feat: add feature'`)
7. Push (`git push origin feature/my-feature`)
8. Open a Pull Request

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
