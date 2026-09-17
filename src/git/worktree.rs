use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default worktree directory relative to project root
pub const DEFAULT_WORKTREE_DIR: &str = ".agtx/worktrees";

/// Create a new git worktree for a task from the detected default branch.
pub fn create_worktree(project_path: &Path, task_slug: &str) -> Result<PathBuf> {
    let base_branch = detect_main_branch(project_path)?;
    create_worktree_from_base(project_path, task_slug, &base_branch, DEFAULT_WORKTREE_DIR)
}

/// Create a new git worktree for a task from the specified base branch.
pub fn create_worktree_from_base(
    project_path: &Path,
    task_slug: &str,
    base_branch: &str,
    worktree_dir: &str,
) -> Result<PathBuf> {
    create_worktree_with_prefix(project_path, task_slug, base_branch, worktree_dir, "task")
}

/// Create a new git worktree for a task with a configurable branch prefix.
pub fn create_worktree_with_prefix(
    project_path: &Path,
    task_slug: &str,
    base_branch: &str,
    worktree_dir: &str,
    branch_prefix: &str,
) -> Result<PathBuf> {
    let worktree_path = project_path.join(worktree_dir).join(task_slug);

    // If worktree already exists and is valid, return it
    if worktree_path.exists() && worktree_path.join(".git").exists() {
        return Ok(worktree_path);
    }

    // Clean up any partial worktree
    if worktree_path.exists() {
        let _ = std::fs::remove_dir_all(&worktree_path);
    }

    // Ensure parent directory exists
    if let Some(parent) = worktree_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let base_branch = resolve_base_branch(project_path, base_branch)?;

    // Create worktree with a new branch based on the requested base branch
    let branch_name = format!("{}/{}", branch_prefix, task_slug);

    // First, try to delete the branch if it exists (from a previous failed attempt)
    let _ = Command::new("git")
        .current_dir(project_path)
        .args(["branch", "-D", &branch_name])
        .output();

    let output = Command::new("git")
        .current_dir(project_path)
        .args(["worktree", "add"])
        .arg(&worktree_path)
        .args(["-b", &branch_name, &base_branch])
        .output()
        .context("Failed to create git worktree")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to create worktree: {}", stderr);
    }

    Ok(worktree_path)
}

fn resolve_base_branch(project_path: &Path, base_branch: &str) -> Result<String> {
    let base_branch = base_branch.trim();
    if base_branch.is_empty() {
        return detect_main_branch(project_path);
    }

    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", base_branch])
        .output()
        .context("Failed to verify configured base branch")?;

    if output.status.success() {
        Ok(base_branch.to_string())
    } else {
        anyhow::bail!("Configured base branch '{}' was not found", base_branch);
    }
}

/// Agent config directories that are always copied from project root to worktrees.
/// These contain commands, skills, and configuration that agents need.
pub const AGENT_CONFIG_DIRS: &[&str] = &[
    ".claude",
    ".gemini",
    ".codex",
    ".github/agents",
    ".config/opencode",
    ".omp",
];

/// Output from a shell script run inside a worktree.
#[derive(Debug)]
pub(crate) struct ScriptOutput {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

/// Run a shell script inside a worktree, capturing stdout/stderr.
pub(crate) fn run_worktree_script(
    script: &str,
    worktree_path: &Path,
    envs: &[(String, String)],
) -> Result<ScriptOutput> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .current_dir(worktree_path)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .output()
        .with_context(|| format!("Failed to run script: {}", script))?;

    Ok(ScriptOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Initialize a worktree by copying agent config dirs, user-specified files, and running an init script.
///
/// Returns a Vec of warning messages for any issues encountered.
/// Does not fail fatally — errors are collected and returned for the caller to display.
pub fn initialize_worktree(
    project_path: &Path,
    worktree_path: &Path,
    copy_files: Option<&str>,
    init_script: Option<&str>,
    copy_dirs: &[String],
) -> Vec<String> {
    let mut warnings = Vec::new();

    // Always copy agent config directories, but never follow symlinks into
    // profile state or credentials outside the project.
    for dir_name in AGENT_CONFIG_DIRS {
        let src = project_path.join(dir_name);
        let Ok(metadata) = std::fs::symlink_metadata(&src) else {
            continue;
        };
        if !metadata.is_dir() && !metadata.file_type().is_symlink() {
            continue;
        }
        match first_symlink_component(project_path, &src) {
            Ok(Some(symlink)) => {
                append_skipped_symlink_warnings(&mut warnings, project_path, vec![symlink]);
                continue;
            }
            Ok(None) => {}
            Err(e) => {
                warnings.push(format!("Failed to inspect '{}': {}", dir_name, e));
                continue;
            }
        }

        let dst = worktree_path.join(dir_name);
        let mut skipped_symlinks = Vec::new();
        if let Err(e) = copy_dir_recursive_impl(&src, &dst, worktree_path, &mut skipped_symlinks) {
            warnings.push(format!("Failed to copy '{}' to worktree: {}", dir_name, e));
        }
        append_skipped_symlink_warnings(&mut warnings, project_path, skipped_symlinks);
    }

    // Copy plugin-specific extra directories
    for dir_name in copy_dirs {
        let src = project_path.join(dir_name);
        if src.is_dir() {
            // Validate path stays within project root
            if let (Ok(canon_proj), Ok(canon_src)) =
                (project_path.canonicalize(), src.canonicalize())
            {
                if !canon_src.starts_with(&canon_proj) {
                    warnings.push(format!(
                        "copy_dirs: '{}' resolves outside project root, skipping (path traversal blocked)",
                        dir_name
                    ));
                    continue;
                }
            }
            match first_symlink_component(project_path, &src) {
                Ok(Some(symlink)) => {
                    append_skipped_symlink_warnings(&mut warnings, project_path, vec![symlink]);
                    continue;
                }
                Ok(None) => {}
                Err(e) => {
                    warnings.push(format!("Failed to inspect '{}': {}", dir_name, e));
                    continue;
                }
            }
            let dst = worktree_path.join(dir_name);
            let mut skipped_symlinks = Vec::new();
            if let Err(e) =
                copy_dir_recursive_impl(&src, &dst, worktree_path, &mut skipped_symlinks)
            {
                warnings.push(format!("Failed to copy '{}' to worktree: {}", dir_name, e));
            }
            append_skipped_symlink_warnings(&mut warnings, project_path, skipped_symlinks);
        }
    }

    // Copy user-specified files/directories
    if let Some(files_str) = copy_files {
        // Pre-compute canonical project path for traversal checks
        let canonical_project = project_path.canonicalize().ok();

        for entry in files_str.split(',') {
            let file_name = entry.trim();
            if file_name.is_empty() {
                continue;
            }

            // Reject obvious traversal patterns before touching the filesystem
            if file_name.contains("..") {
                warnings.push(format!(
                    "copy_files: '{}' contains '..', skipping (path traversal blocked)",
                    file_name
                ));
                continue;
            }

            let src = project_path.join(file_name);
            let dst = worktree_path.join(file_name);

            let metadata = match std::fs::symlink_metadata(&src) {
                Ok(metadata) => metadata,
                Err(_) => {
                    warnings.push(format!(
                        "copy_files: '{}' not found in project root, skipping",
                        file_name
                    ));
                    continue;
                }
            };
            // Validate the resolved target before the generic symlink warning so
            // links escaping the project retain the actionable traversal reason.
            if let Some(ref canon_proj) = canonical_project {
                if let Ok(canon_src) = src.canonicalize() {
                    if !canon_src.starts_with(canon_proj) {
                        warnings.push(format!(
                            "copy_files: '{}' resolves outside project root, skipping (path traversal blocked)",
                            file_name
                        ));
                        continue;
                    }
                }
            }
            match first_symlink_component(project_path, &src) {
                Ok(Some(symlink)) => {
                    append_skipped_symlink_warnings(&mut warnings, project_path, vec![symlink]);
                    continue;
                }
                Ok(None) => {}
                Err(e) => {
                    warnings.push(format!("Failed to inspect '{}': {}", file_name, e));
                    continue;
                }
            }

            if metadata.is_dir() {
                let mut skipped_symlinks = Vec::new();
                if let Err(e) =
                    copy_dir_recursive_impl(&src, &dst, worktree_path, &mut skipped_symlinks)
                {
                    warnings.push(format!(
                        "Failed to copy directory '{}' to worktree: {}",
                        file_name, e
                    ));
                }
                append_skipped_symlink_warnings(&mut warnings, project_path, skipped_symlinks);
            } else {
                if let Some(parent) = dst.parent() {
                    if let Err(e) = create_destination_dir(worktree_path, parent) {
                        warnings.push(format!(
                            "Failed to create directory for '{}': {}",
                            file_name, e
                        ));
                        continue;
                    }
                }
                if let Err(e) = ensure_destination_path_safe(worktree_path, &dst) {
                    warnings.push(format!("Failed to copy '{}' to worktree: {}", file_name, e));
                    continue;
                }
                if let Err(e) = std::fs::copy(&src, &dst) {
                    warnings.push(format!("Failed to copy '{}' to worktree: {}", file_name, e));
                }
            }
        }
    }

    if let Some(script) = init_script {
        let script = script.trim();
        if !script.is_empty() {
            tracing::info!(
                script = script,
                worktree = %worktree_path.display(),
                "Executing project init_script"
            );
            match run_worktree_script(script, worktree_path, &[]) {
                Ok(result) => {
                    if !result.status.success() {
                        warnings.push(format!(
                            "init_script exited with {}: {}",
                            result.status,
                            result.stderr.trim()
                        ));
                    }
                }
                Err(e) => warnings.push(format!("Failed to run init_script: {}", e)),
            }
        }
    }

    warnings
}

fn append_skipped_symlink_warnings(
    warnings: &mut Vec<String>,
    project_path: &Path,
    skipped_symlinks: Vec<PathBuf>,
) {
    for skipped in skipped_symlinks {
        let relative = skipped.strip_prefix(project_path).unwrap_or(&skipped);
        warnings.push(format!(
            "Skipped symlink '{}' while copying to worktree",
            relative.display()
        ));
    }
}

/// Return the first symlink between `root` and `path`, including `path`.
///
/// Looking only at `symlink_metadata(path)` misses an intermediate alias in a
/// configured path such as `alias/config.toml`.
fn first_symlink_component(root: &Path, path: &Path) -> Result<Option<PathBuf>> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("'{}' is outside '{}'", path.display(), root.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(name) => current.push(name),
            std::path::Component::CurDir => continue,
            _ => anyhow::bail!("path '{}' escapes '{}'", path.display(), root.display()),
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => return Ok(Some(current)),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(None)
}

/// Reject any existing symlink at or below the trusted destination root.
fn ensure_destination_path_safe(root: &Path, path: &Path) -> Result<()> {
    let relative = path.strip_prefix(root).with_context(|| {
        format!(
            "destination '{}' is outside '{}'",
            path.display(),
            root.display()
        )
    })?;

    if std::fs::symlink_metadata(root)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        anyhow::bail!("destination root '{}' is a symlink", root.display());
    }

    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(name) => current.push(name),
            std::path::Component::CurDir => continue,
            _ => anyhow::bail!(
                "destination path '{}' escapes '{}'",
                path.display(),
                root.display()
            ),
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                anyhow::bail!("destination path '{}' is a symlink", current.display())
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn create_destination_dir(root: &Path, path: &Path) -> Result<()> {
    ensure_destination_path_safe(root, path)?;
    std::fs::create_dir_all(path)?;
    // Inspect every component that now exists before writing a child through it.
    ensure_destination_path_safe(root, path)
}

/// Recursively copy a directory and its contents without following symlinks.
pub fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    let mut skipped_symlinks = Vec::new();
    let destination_root = dst.parent().unwrap_or(dst);
    copy_dir_recursive_impl(src, dst, destination_root, &mut skipped_symlinks)?;
    if !skipped_symlinks.is_empty() {
        anyhow::bail!(
            "skipped {} symlink(s) while recursively copying '{}'",
            skipped_symlinks.len(),
            src.display()
        );
    }
    Ok(())
}

fn copy_dir_recursive_impl(
    src: &Path,
    dst: &Path,
    destination_root: &Path,
    skipped_symlinks: &mut Vec<PathBuf>,
) -> Result<()> {
    let metadata = std::fs::symlink_metadata(src)?;
    if metadata.file_type().is_symlink() {
        skipped_symlinks.push(src.to_path_buf());
        return Ok(());
    }

    create_destination_dir(destination_root, dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&src_path)?;
        if metadata.file_type().is_symlink() {
            skipped_symlinks.push(src_path);
        } else if metadata.is_dir() {
            copy_dir_recursive_impl(&src_path, &dst_path, destination_root, skipped_symlinks)?;
        } else {
            ensure_destination_path_safe(destination_root, &dst_path)?;
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Detect the main branch name (main or master)
pub fn detect_main_branch(project_path: &Path) -> Result<String> {
    // Check if 'main' exists
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", "main"])
        .output()
        .context("Failed to check for main branch")?;

    if output.status.success() {
        return Ok("main".to_string());
    }

    // Check if 'master' exists
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", "master"])
        .output()
        .context("Failed to check for master branch")?;

    if output.status.success() {
        return Ok("master".to_string());
    }

    // Fallback: get the current branch.
    //
    // The exit status matters here and reading stdout alone is not enough. On an
    // unborn branch — a fresh `git init` with no commits, which is exactly how a
    // greenfield project starts — this fails with 128 *and still prints the
    // literal string* `HEAD`, which is not a revision a worktree can be cut from
    // (`git worktree add` answers `invalid reference: HEAD`).
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("Failed to get current branch")?;

    if !output.status.success() {
        // Distinguish "no commits yet" from any other failure before acting: the
        // recovery below writes to the user's repository, and it is only
        // unambiguously safe on a repo that has no history to disturb.
        if has_no_commits(project_path) {
            create_initial_commit(project_path)?;
            return current_branch_name(project_path);
        }
        anyhow::bail!(
            "Could not determine the base branch: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Whether the repository has no commits at all (HEAD is unborn).
fn has_no_commits(project_path: &Path) -> bool {
    Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(false)
}

/// Give an empty repository the one commit a worktree needs to branch from.
///
/// A worktree must be cut from a commit, so a repository with no history cannot
/// host a task at all. Refusing would be defensible, but "run `git commit
/// --allow-empty` and start again" is the only answer to that refusal, and
/// making the user type it buys nothing: an empty commit on a repo with no
/// history discards nothing and conflicts with nothing.
///
/// Deliberately narrow. This runs only when [`has_no_commits`] holds, so it can
/// never rewrite, amend or displace work that already exists.
fn create_initial_commit(project_path: &Path) -> Result<()> {
    tracing::info!(
        project = %project_path.display(),
        "Repository has no commits; creating an empty initial commit so a worktree can be cut"
    );
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["commit", "--allow-empty", "-m", "init"])
        .output()
        .context("Failed to create the initial commit")?;

    if !output.status.success() {
        anyhow::bail!(
            "Repository has no commits and one could not be created: {}. \
             Make a commit first, then retry.",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// The current branch name, after HEAD is known to exist.
fn current_branch_name(project_path: &Path) -> Result<String> {
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("Failed to get current branch")?;

    if !output.status.success() {
        anyhow::bail!(
            "Could not determine the current branch: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// True when `path` is a repository's *main* working tree rather than a linked
/// worktree.
///
/// This is the shape `skip_worktree` produces: a task's `worktree_path` is the
/// user's own checkout, and every cleanup path then asks for that to be removed.
/// git already refuses ("is a main working tree"), so today the protection is
/// inherited rather than intended — and `fs::rename`, which a background trash
/// would use instead, has no such concept.
///
/// Two conditions, and both are needed:
///
/// - `--git-dir` == `--git-common-dir`. A linked worktree's git dir is
///   `{repo}/.git/worktrees/{name}` while its common dir is `{repo}/.git`.
/// - `--show-toplevel` is `path` itself.
///
/// The second is not redundant. The default `worktree_dir` is `.agtx/worktrees`,
/// *inside* the project, so a worktree there that has lost its `.git` link makes
/// git walk up to the main repository and report the first condition as true.
/// Without the toplevel check, exactly the half-deleted worktrees that most need
/// removing would be refused as if they were the user's checkout.
pub fn is_main_working_tree(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let rev_parse = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .current_dir(path)
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    };

    let common = rev_parse(&["rev-parse", "--path-format=absolute", "--git-common-dir"]);
    let dir = rev_parse(&["rev-parse", "--path-format=absolute", "--git-dir"]);
    match (common, dir) {
        (Some(common), Some(dir)) if common == dir => {}
        // A linked worktree, not a repository at all, or a git too old for
        // --path-format. Not provably a main working tree, so this does not
        // block the removal; the caller's project-root comparison is the second
        // line of defence.
        _ => return false,
    }

    let Some(toplevel) = rev_parse(&["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    match (Path::new(&toplevel).canonicalize(), path.canonicalize()) {
        (Ok(top), Ok(this)) => top == this,
        _ => false,
    }
}

/// Get the worktree path for a task
pub fn worktree_path(project_path: &Path, task_id: &str, worktree_dir: &str) -> PathBuf {
    project_path.join(worktree_dir).join(task_id)
}

/// Get the worktree path for a task using a custom worktree directory
pub fn worktree_path_with_dir(project_path: &Path, task_id: &str, worktree_dir: &str) -> PathBuf {
    worktree_path(project_path, task_id, worktree_dir)
}

/// Check if a worktree exists for a task
pub fn worktree_exists(project_path: &Path, task_id: &str) -> bool {
    worktree_path(project_path, task_id, DEFAULT_WORKTREE_DIR).exists()
}

/// Check if a worktree exists for a task using a custom worktree directory
pub fn worktree_exists_with_dir(project_path: &Path, task_id: &str, worktree_dir: &str) -> bool {
    worktree_path_with_dir(project_path, task_id, worktree_dir).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_run_worktree_script_captures_output_and_env() {
        let temp_dir = TempDir::new().unwrap();
        let envs = vec![("AGTX_TASK_ID".to_string(), "task-123".to_string())];

        let output = run_worktree_script("echo $AGTX_TASK_ID", temp_dir.path(), &envs).unwrap();

        assert!(output.status.success());
        assert_eq!(output.stdout.trim(), "task-123");
    }

    #[test]
    fn test_run_worktree_script_nonzero_exit() {
        let temp_dir = TempDir::new().unwrap();

        let output = run_worktree_script("exit 42", temp_dir.path(), &[]).unwrap();

        assert!(!output.status.success());
    }
}
