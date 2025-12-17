use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

fn vgrep() -> Command {
    Command::cargo_bin("vgrep").unwrap()
}

#[test]
fn test_help() {
    vgrep()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("semantic grep"));
}

#[test]
fn test_version() {
    vgrep()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("vgrep"));
}

#[test]
fn test_init() {
    let temp_dir = TempDir::new().unwrap();

    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialized"));

    // Check that .vgrep directory was created
    assert!(temp_dir.path().join(".vgrep").exists());
    assert!(temp_dir.path().join(".vgrep/config.json").exists());
}

#[test]
fn test_init_twice_fails() {
    let temp_dir = TempDir::new().unwrap();

    // First init
    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Second init without --force should not re-initialize
    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("already initialized"));
}

#[test]
fn test_init_force() {
    let temp_dir = TempDir::new().unwrap();

    // First init
    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Second init with --force should work
    vgrep()
        .args(["init", "--force"])
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialized"));
}

#[test]
fn test_status_without_init() {
    let temp_dir = TempDir::new().unwrap();

    vgrep()
        .arg("status")
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("not initialized"));
}

#[test]
fn test_status_after_init() {
    let temp_dir = TempDir::new().unwrap();

    // Initialize
    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Check status
    vgrep()
        .arg("status")
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed files: 0"));
}

#[test]
fn test_models_info() {
    vgrep()
        .args(["models", "info"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Qwen3-Embedding"));
}

#[test]
fn test_models_list() {
    let temp_dir = TempDir::new().unwrap();

    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    vgrep()
        .args(["models", "list"])
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Embedding"));
}

#[test]
fn test_config_show() {
    let temp_dir = TempDir::new().unwrap();

    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    vgrep()
        .args(["config", "--show"])
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("chunk_size"));
}

#[test]
fn test_index_without_model() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test file
    fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();

    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Index should fail without model
    vgrep()
        .arg("index")
        .current_dir(temp_dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("model not found"));
}

#[test]
fn test_search_without_model() {
    let temp_dir = TempDir::new().unwrap();

    vgrep()
        .arg("init")
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Search should fail without model
    vgrep()
        .args(["search", "test query"])
        .current_dir(temp_dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("model not found"));
}
