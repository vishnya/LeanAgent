import yaml
import re
from pathlib import Path

TASKFILE = Path(__file__).resolve().parent.parent / "Taskfile.yaml"

# ---------------------------------------------------------------------------
# Define the set of *allowed* tasks. If you intentionally add or rename a
# task in Taskfile.yaml, you **must** update this list so the test suite
# continues to pass.
# ---------------------------------------------------------------------------
ALLOWED_TASKS = {
    "test",
    "check_github_token",
    "create_conda_env",
    "install_elan",
    "install_requirements",
    "download_checkpoint_data",
    "replace_files",
    "setup",
    "run",
    "run_fisher",
    "run_lifelong_learning",
}


def _read_taskfile_yaml() -> dict:
    """Load Taskfile.yaml as Python dict using safe YAML parsing."""
    assert TASKFILE.is_file(), "Taskfile.yaml missing"
    with TASKFILE.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_tasks_match_allowed_list():
    """Ensure Taskfile.yaml defines *exactly* the allowed tasks."""
    content = _read_taskfile_yaml()
    tasks = set(content.get("tasks", {}).keys())
    extra = tasks - ALLOWED_TASKS
    missing = ALLOWED_TASKS - tasks
    assert not extra, f"Found unsupported tasks in Taskfile.yaml: {sorted(extra)}"
    assert not missing, f"Expected tasks missing from Taskfile.yaml: {sorted(missing)}"


def test_no_unresolved_python_placeholders():
    """Ensure we don't leave {raid_dir} style placeholders inside Taskfile (Go templates use {{.}})."""
    text = TASKFILE.read_text(encoding="utf-8")
    unresolved = re.findall(r"\{[a-zA-Z_]+\}", text)
    # Allow curly braces used in bash parameter expansion e.g., ${RAY_TMPDIR}
    unresolved = [u for u in unresolved if not u.startswith("${")]
    assert not unresolved, f"Found unresolved placeholders: {unresolved}"
