#!/usr/bin/env python3
"""
Environment & Setup Verification Script for SE373 - Topic 05: Interview Agent
"""

import shutil
import subprocess
import sys
from pathlib import Path


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major >= 3 and v.minor >= 11:
        return True, f"Python {ver_str} (>= 3.11)"
    return False, f"Python {ver_str} (< 3.11 requirement)"


def check_imports() -> tuple[bool, str]:
    required_modules = [
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("pydantic", "pydantic"),
        ("chromadb", "chromadb"),
        ("fitz (PyMuPDF)", "fitz"),
        ("rank_bm25", "rank_bm25"),
    ]
    failed = []
    success = []
    for display_name, mod_name in required_modules:
        try:
            __import__(mod_name)
            success.append(display_name)
        except ImportError as e:
            failed.append(f"{display_name} ({e})")

    if failed:
        return False, f"Missing: {', '.join(failed)}"
    return True, f"All modules imported: {', '.join(success)}"


def check_directories() -> tuple[bool, str]:
    required_dirs = [
        "data/knowledge/competencies",
        "data/knowledge/questions",
        "data/knowledge/rubrics",
        "data/knowledge/coding_tasks",
        "data/knowledge/policies",
        "evals/data/jds",
        "evals/data/cvs",
        "evals/data/counterfactual_pairs",
        "evals/data/gold_answers",
        "evals/data/adversarial_cases",
        "skills/coding-assessment/references",
        "skills/coding-assessment/scripts",
        "skills/rubric-grading/references",
        "skills/rubric-grading/scripts",
        "traces",
        "docs",
    ]
    missing = [d for d in required_dirs if not Path(d).is_dir()]
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, f"All {len(required_dirs)} required directories exist"


def check_files() -> tuple[bool, str]:
    required_files = [
        ".env.example",
        ".gitignore",
        ".pre-commit-config.yaml",
        "AGENTS.md",
    ]
    missing = [f for f in required_files if not Path(f).is_file()]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, f"All {len(required_files)} configuration/doc files exist"


def check_eval_runner() -> tuple[bool, str]:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "evals.run", "--all"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return True, "python -m evals.run --all exited with code 0"
        return (
            False,
            f"evals.run failed with code {res.returncode}: {res.stderr.strip()}",
        )
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"Failed to execute evals.run: {e}"


def check_docker() -> tuple[bool, str]:
    if not shutil.which("docker"):
        return True, "Docker CLI not found (Skipped)"

    # Check if Docker daemon is running
    try:
        info_res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if info_res.returncode != 0:
            return True, "Docker daemon not running (Skipped sandbox container check)"
    except (subprocess.SubprocessError, OSError):
        return True, "Docker daemon not reachable (Skipped sandbox container check)"

    # If daemon is running, verify the sandbox image
    image_name = "se373-interview-sandbox:latest"
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--user",
            "10001:10001",
            image_name,
            "python",
            "-c",
            "import sys; print(f'Sandbox Python {sys.version.split()[0]} OK')",
        ]
        test_run = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if test_run.returncode == 0:
            return (
                True,
                f"Image '{image_name}' verified with isolated flags (--network none, 256m, 0.5 cpu, non-root)",
            )
        return False, f"Sandbox container run failed: {test_run.stderr.strip()}"
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"Error verifying sandbox container: {e}"


def main() -> int:
    checks = [
        ("Python Version (>= 3.11)", check_python_version),
        ("Required Packages Import", check_imports),
        ("Required Directory Tree", check_directories),
        ("Config & Agent-Readable Docs", check_files),
        ("Eval Runner Entrypoint", check_eval_runner),
        ("Docker Sandbox (UC-04)", check_docker),
    ]

    print("\n" + "=" * 70)
    print(" SE373 - Topic 05: Interview Agent | Environment Verification")
    print("=" * 70)

    results = []
    all_passed = True

    for name, check_fn in checks:
        passed, msg = check_fn()
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        results.append((name, status, msg))

    print(f"{'Check Name':<32} | {'Status':<8} | {'Details'}")
    print("-" * 70)
    for name, status, msg in results:
        print(f"{name:<32} | {status:<8} | {msg}")
    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL ENVIRONMENT VERIFICATION CHECKS PASSED!\n")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED. Please review the details above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
