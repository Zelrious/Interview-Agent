"""
tests/test_github_backlog.py

Kiểm thử tự động cho cấu trúc Backlog (docs/github_tasks_backlog.json)
và script quản trị đồng bộ GitHub (scripts/manage_github_tasks.py).
"""

from pathlib import Path

from scripts.manage_github_tasks import (
    detect_github_repo,
    filter_tasks_by_phase,
    load_backlog,
    parse_user_map,
)


def test_backlog_json_integrity() -> None:
    backlog_path = Path("docs/github_tasks_backlog.json")
    assert backlog_path.is_file(), "File docs/github_tasks_backlog.json phải tồn tại"

    data = load_backlog(backlog_path)
    assert len(data["milestones"]) == 4, "Phải có đúng 4 Milestones/Phases"
    assert len(data["labels"]) == 20, "Phải có đúng 20 Labels chuẩn hóa"
    assert len(data["tasks"]) == 12, "Phải có đúng 12 Tasks"

    # Kiểm tra tính hợp lệ của từng Milestone
    for m in data["milestones"]:
        assert "number" in m
        assert "title" in m
        assert "due_on" in m
        assert m["state"] == "open"

    # Kiểm tra danh mục nhãn
    label_names = {lbl["name"] for lbl in data["labels"]}
    assert "K1:Model-Loop" in label_names
    assert "K3:RAG" in label_names
    assert "P0:Blocker" in label_names
    assert "P1:Critical" in label_names
    assert "RAG-Buoi06" in label_names

    # Kiểm tra từng task
    valid_roles = {"SV1", "SV2", "SV3", "SV4", "SV5"}
    for t in data["tasks"]:
        assert t["task_id"], f"Task {t} thiếu task_id"
        assert t["title"].startswith(f"[{t['task_id']}]")
        assert t["owner_role"] in valid_roles, f"Role {t['owner_role']} không hợp lệ"
        assert len(t["labels"]) >= 3, f"Task {t['task_id']} phải có ít nhất 3 nhãn"
        for label in t["labels"]:
            assert (
                label in label_names
            ), f"Nhãn {label} chưa được khai báo trong danh mục labels"
        body = t["body"]
        assert "### U1" in body
        assert "### U2" in body
        assert "### U3" in body
        assert "### Khai báo Minh bạch Sử dụng AI" in body


def test_filter_tasks_by_phase() -> None:
    backlog_path = Path("docs/github_tasks_backlog.json")
    data = load_backlog(backlog_path)
    all_tasks = data["tasks"]

    phase1_tasks = filter_tasks_by_phase(all_tasks, "1")
    assert (
        len(phase1_tasks) == 5
    ), f"Phase 1 phải có 5 tasks, thực tế có {len(phase1_tasks)}"
    for t in phase1_tasks:
        assert "Phase 1" in t["milestone"]

    phase2_tasks = filter_tasks_by_phase(all_tasks, "2")
    assert (
        len(phase2_tasks) == 3
    ), f"Phase 2 phải có 3 tasks, thực tế có {len(phase2_tasks)}"

    phase3_tasks = filter_tasks_by_phase(all_tasks, "3")
    assert (
        len(phase3_tasks) == 4
    ), f"Phase 3 phải có 4 tasks, thực tế có {len(phase3_tasks)}"

    all_filtered = filter_tasks_by_phase(all_tasks, "all")
    assert len(all_filtered) == 12


def test_parse_user_map() -> None:
    raw = "sv1=alice, sv2=bob,sv3=carol"
    res = parse_user_map(raw)
    assert res == {"sv1": "alice", "sv2": "bob", "sv3": "carol"}

    empty_res = parse_user_map(None)
    assert empty_res == {}


def test_detect_github_repo() -> None:
    override_repo = detect_github_repo("myorg/myrepo")
    assert override_repo == "myorg/myrepo"

    auto_repo = detect_github_repo(None)
    assert "Zelrious/Interview-Agent" in auto_repo or "/" in auto_repo
