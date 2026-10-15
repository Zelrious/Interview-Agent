#!/usr/bin/env python3
"""
scripts/manage_github_tasks.py

Công cụ quản lý Task, Phase (Milestone), Labels và Issue Synchronization trên GitHub
theo chuẩn bảng dự án mẫu phhaifit/projects/2 và Rubric v3 của môn học SE373 (Đề tài 05).

Chức năng:
1. --dry-run : Hiển thị bảng tổng hợp toàn bộ 4 Phases, 20 Labels và 12 Tasks từ docs/github_tasks_backlog.json.
2. --push    : Đồng bộ tự động Milestones, Labels và đẩy Issues lên GitHub Repository
               (hỗ trợ lọc --phase 1..4 hoặc all, và gán assignees qua --map-users sv1=user1,sv2=user2...).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def detect_github_repo(cli_repo: str | None = None) -> str:
    """Tự động nhận diện GitHub repo (owner/repo) từ git remote hoặc tham số CLI."""
    if cli_repo:
        return cli_repo.strip()

    try:
        res = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = res.stdout.strip()
        match = re.search(
            r"github\.com[/:](?P<owner>[^/]+)/(?P<repo>[^/\s]+?)(?:\.git)?$", url
        )
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return "Zelrious/Interview-Agent"


def load_backlog(backlog_path: Path) -> dict[str, Any]:
    """Đọc và kiểm tra tính hợp lệ của file JSON backlog."""
    if not backlog_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file backlog tại: {backlog_path}")

    with open(backlog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "milestones" not in data or "labels" not in data or "tasks" not in data:
        raise ValueError(
            "File backlog thiếu một trong các khóa: 'milestones', 'labels', 'tasks'"
        )

    return data


def parse_user_map(map_str: str | None) -> dict[str, str]:
    """Parse chuỗi mapping sv1=user1,sv2=user2 thành dict."""
    user_map: dict[str, str] = {}
    if not map_str:
        return user_map

    for pair in map_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            role, user = pair.split("=", 1)
            user_map[role.strip().lower()] = user.strip()
    return user_map


def filter_tasks_by_phase(
    tasks: list[dict[str, Any]], phase_arg: str
) -> list[dict[str, Any]]:
    """Lọc danh sách tasks theo phase được chọn."""
    if phase_arg.lower() in ("all", "0", ""):
        return tasks

    target = phase_arg.lower().replace("-", " ").strip()
    if target.isdigit():
        target = f"phase {target}"

    filtered = []
    for t in tasks:
        milestone = t.get("milestone", "").lower()
        if target in milestone:
            filtered.append(t)
    return filtered


def run_dry_run(
    backlog: dict[str, Any], phase_filter: str, user_map: dict[str, str]
) -> None:
    """In bảng tổng quan chi tiết về Milestones, Labels và Tasks lên terminal."""
    project_name = backlog.get("project_name", "SE373 — Đề tài 05: Interview Agent")
    milestones = backlog.get("milestones", [])
    labels = backlog.get("labels", [])
    tasks = backlog.get("tasks", [])
    filtered_tasks = filter_tasks_by_phase(tasks, phase_filter)

    print("\n" + "=" * 90)
    print(f" 📋 {project_name.upper()} — GITHUB TASK MANAGEMENT")
    print("=" * 90)
    print(f"• Tổng số Milestones (Phases) : {len(milestones)}")
    print(f"• Tổng số Labels chuẩn hóa    : {len(labels)}")
    print(
        f"• Tổng số Tasks trong Backlog : {len(tasks)} (Đang xem: {len(filtered_tasks)} tasks cho Phase '{phase_filter}')"
    )
    if user_map:
        mapped_desc = ", ".join(f"{k.upper()} -> @{v}" for k, v in user_map.items())
        print(f"• Phân vai thành viên         : {mapped_desc}")
    print("-" * 90)

    # 1. BẢNG MILESTONES
    print("\n🎯 [DANH MỤC MILESTONES / PHASES]")
    print(f"{'#':<3} | {'Milestone Title':<68} | {'Hạn chót (Due)':<12}")
    print("-" * 90)
    for m in milestones:
        due = m.get("due_on", "")[:10]
        print(f"#{m.get('number', '-'):<2} | {m.get('title', ''):<68} | {due:<12}")

    # 2. BẢNG LABELS
    print("\n🏷️  [DANH MỤC LABELS CHUẨN HOÁ]")
    k_labels = [lbl for lbl in labels if lbl.get("name", "").startswith("K")]
    p_labels = [lbl for lbl in labels if lbl.get("name", "").startswith("P")]
    uc_labels = [
        lbl
        for lbl in labels
        if not lbl.get("name", "").startswith("K")
        and not lbl.get("name", "").startswith("P")
    ]

    def print_label_group(group_name: str, group_labels: list[dict[str, Any]]) -> None:
        print(f"  • {group_name}:")
        for lbl in group_labels:
            color = f"#{lbl.get('color', '')}"
            name = lbl.get("name", "")
            desc = lbl.get("description", "")
            print(f"    - [{name:<18}] ({color}) : {desc}")

    print_label_group("Mảng Kiến thức (K1–K8)", k_labels)
    print_label_group("Mức Ưu tiên (Priority)", p_labels)
    print_label_group("Use Case & Safety Tags", uc_labels)

    # 3. BẢNG TASKS CHI TIẾT
    print(f"\n📌 [DANH SÁCH TASKS CHI TIẾT — {len(filtered_tasks)} TASKS]")
    print("=" * 90)
    for idx, t in enumerate(filtered_tasks, 1):
        task_id = t.get("task_id", f"TASK-{idx:02d}")
        owner_role = t.get("owner_role", "Chưa gán")
        mapped_user = user_map.get(owner_role.lower())
        assignee_str = f"@{mapped_user}" if mapped_user else f"({owner_role})"
        labels_str = ", ".join(t.get("labels", []))

        print(f"\n[{idx:02d}] {t.get('title')}")
        print(
            f"     ID: {task_id} | Người phụ trách: {assignee_str} | Milestone: {t.get('milestone')}"
        )
        print(f"     Labels: [{labels_str}]")

        body = t.get("body", "")
        u_lines = []
        for line in body.splitlines():
            line_strip = line.strip()
            if line_strip.startswith(("### U", "### Khai báo")):
                u_lines.append(f"     {line_strip}")
            elif line_strip.startswith(("- **Nhiệm vụ:**", "- **DoD:**")):
                u_lines.append(f"       {line_strip}")
        if u_lines:
            for ul in u_lines[:6]:
                print(ul)
            if len(u_lines) > 6:
                print("       ...")
        print("-" * 90)

    print("\n💡 GỢI Ý ĐỒNG BỘ LÊN GITHUB:")
    print("  Để đẩy toàn bộ hoặc Phase 1 lên GitHub, chạy lệnh:")
    print(
        "  python scripts/manage_github_tasks.py --push --phase 1 --map-users sv1=github_user1,sv2=github_user2\n"
    )


class GitHubClient:
    """GitHub REST API Client đơn giản sử dụng urllib tiêu chuẩn."""

    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo}"

    def _request(
        self, method: str, endpoint: str, data: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SE373-Interview-Agent-TaskManager",
        }
        body_bytes = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url, data=body_bytes, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req) as response:
                resp_text = response.read().decode("utf-8")
                if resp_text:
                    return json.loads(resp_text)
                return None
        except urllib.error.HTTPError as e:
            err_content = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"GitHub API Error {e.code} on {method} {url}: {err_content}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Request failed on {method} {url}: {e}") from e

    def get_milestones(self) -> list[dict[str, Any]]:
        return self._request("GET", "milestones?state=all&per_page=100") or []

    def create_milestone(
        self, title: str, description: str, due_on: str
    ) -> dict[str, Any]:
        payload = {"title": title, "description": description, "state": "open"}
        if due_on:
            payload["due_on"] = due_on
        return self._request("POST", "milestones", payload)

    def get_labels(self) -> list[dict[str, Any]]:
        return self._request("GET", "labels?per_page=100") or []

    def create_label(self, name: str, color: str, description: str) -> dict[str, Any]:
        payload = {"name": name, "color": color.lstrip("#"), "description": description}
        return self._request("POST", "labels", payload)

    def get_issues(self) -> list[dict[str, Any]]:
        return self._request("GET", "issues?state=all&per_page=100") or []

    def create_issue(
        self,
        title: str,
        body: str,
        milestone_number: int | None,
        labels: list[str],
        assignees: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": title,
            "body": body,
            "labels": labels,
        }
        if milestone_number is not None:
            payload["milestone"] = milestone_number
        if assignees:
            payload["assignees"] = assignees
        return self._request("POST", "issues", payload)


def resolve_token(cli_token: str | None) -> str | None:
    """Lấy token từ CLI argument, biến môi trường GITHUB_TOKEN / GH_TOKEN hoặc file .env."""
    if cli_token:
        return cli_token.strip()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()

    env_path = Path(".env")
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(("GITHUB_TOKEN=", "GH_TOKEN=")):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass

    return None


def run_push(
    backlog: dict[str, Any],
    repo: str,
    phase_filter: str,
    user_map: dict[str, str],
    token: str,
) -> None:
    """Đồng bộ Milestones, Labels và đẩy Issues lên GitHub qua REST API."""
    print("\n" + "=" * 80)
    print(f"🚀 BẮT ĐẦU ĐỒNG BỘ LÊN GITHUB: {repo}")
    print("=" * 80)

    client = GitHubClient(repo=repo, token=token)

    # 1. ĐỒNG BỘ MILESTONES
    print("\n1️⃣  Đang kiểm tra và đồng bộ Milestones...")
    existing_milestones = client.get_milestones()
    milestone_map: dict[str, int] = {
        m["title"]: m["number"] for m in existing_milestones
    }

    for m in backlog.get("milestones", []):
        m_title = m["title"]
        if m_title in milestone_map:
            print(f"  [ĐÃ CÓ] Milestone #{milestone_map[m_title]}: {m_title}")
        else:
            created = client.create_milestone(
                title=m_title,
                description=m.get("description", ""),
                due_on=m.get("due_on", ""),
            )
            milestone_map[m_title] = created["number"]
            print(f"  [TẠO MỚI] Milestone #{created['number']}: {m_title}")

    # 2. ĐỒNG BỘ LABELS
    print("\n2️⃣  Đang kiểm tra và đồng bộ Labels...")
    existing_labels = client.get_labels()
    existing_label_names = {lbl["name"].lower() for lbl in existing_labels}

    for lbl in backlog.get("labels", []):
        name = lbl["name"]
        if name.lower() in existing_label_names:
            print(f"  [ĐÃ CÓ] Label: {name}")
        else:
            try:
                client.create_label(
                    name=name,
                    color=lbl.get("color", "cccccc"),
                    description=lbl.get("description", ""),
                )
                print(f"  [TẠO MỚI] Label: {name} (#{lbl.get('color')})")
            except (RuntimeError, urllib.error.URLError) as e:
                print(f"  [LỖI TẠO LABEL {name}]: {e}")

    # 3. ĐỒNG BỘ ISSUES (TASKS)
    tasks = backlog.get("tasks", [])
    filtered_tasks = filter_tasks_by_phase(tasks, phase_filter)
    print(
        f"\n3️⃣  Đang kiểm tra và đồng bộ Issues ({len(filtered_tasks)} tasks cho Phase '{phase_filter}')..."
    )

    existing_issues = client.get_issues()
    existing_issue_titles = {iss["title"].strip() for iss in existing_issues}

    created_count = 0
    skipped_count = 0

    for t in filtered_tasks:
        title = t["title"].strip()
        if title in existing_issue_titles:
            print(f"  [BỎ QUA - ĐÃ TỒN TẠI] {title}")
            skipped_count += 1
            continue

        m_title = t.get("milestone", "")
        m_number = milestone_map.get(m_title)

        owner_role = t.get("owner_role", "").lower()
        assignees = []
        if owner_role in user_map:
            assignees.append(user_map[owner_role])

        labels = t.get("labels", [])
        body = t.get("body", "")

        try:
            created_iss = client.create_issue(
                title=title,
                body=body,
                milestone_number=m_number,
                labels=labels,
                assignees=assignees,
            )
            iss_num = created_iss.get("number")
            iss_url = created_iss.get("html_url", "")
            assignee_msg = f" (gán: @{assignees[0]})" if assignees else ""
            print(
                f"  [TẠO THÀNH CÔNG #{iss_num}] {title}{assignee_msg}\n    Link: {iss_url}"
            )
            existing_issue_titles.add(title)
            created_count += 1
        except (RuntimeError, urllib.error.URLError) as e:
            print(f"  [LỖI TẠO ISSUE '{title}']: {e}")

    print("\n" + "=" * 80)
    print(
        f"✅ HOÀN TẤT ĐỒNG BỘ: Tạo mới {created_count} Issues | Bỏ qua trùng lặp: {skipped_count} Issues."
    )
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SE373 Đề tài 05 — Script quản lý và đồng bộ Tasks, Milestones, Labels lên GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In bảng tổng hợp toàn bộ Phases, Labels và 12 Tasks từ backlog lên terminal (mặc định)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Đồng bộ tự động Milestones, Labels và đẩy Issues lên GitHub Repository",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Lọc task theo Phase: '1', '2', '3', '4' hoặc 'all' (Mặc định: 'all')",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Tên GitHub repository dạng 'owner/repo' (Mặc định: tự động phát hiện từ git origin)",
    )
    parser.add_argument(
        "--map-users",
        default=None,
        help="Ánh xạ vai trò SV sang tài khoản GitHub. Ví dụ: 'sv1=user1,sv2=user2,sv3=user3,sv4=user4,sv5=user5'",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub Personal Access Token (nếu không cung cấp sẽ đọc từ GITHUB_TOKEN trong env)",
    )
    parser.add_argument(
        "--backlog",
        default="docs/github_tasks_backlog.json",
        help="Đường dẫn file backlog JSON (Mặc định: docs/github_tasks_backlog.json)",
    )

    args = parser.parse_args()

    backlog_path = Path(args.backlog)
    try:
        backlog = load_backlog(backlog_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as e:
        print(f"❌ Lỗi tải file backlog: {e}", file=sys.stderr)
        sys.exit(1)

    user_map = parse_user_map(args.map_users)
    repo = detect_github_repo(args.repo)

    if not args.push:
        run_dry_run(backlog, phase_filter=args.phase, user_map=user_map)
        return

    token = resolve_token(args.token)
    if not token:
        print(
            "\n❌ LỖI: Không tìm thấy GitHub Access Token để thực hiện thao tác --push!",
            file=sys.stderr,
        )
        print("  Vui lòng cung cấp token qua một trong các cách sau:")
        print(
            "  1. Thiết lập biến môi trường: export GITHUB_TOKEN='ghp_xxx' (Linux/macOS) hoặc $env:GITHUB_TOKEN='ghp_xxx' (PowerShell)"
        )
        print(
            "  2. Truyền tham số CLI: python scripts/manage_github_tasks.py --push --token 'ghp_xxx'"
        )
        print("  3. Đặt dòng 'GITHUB_TOKEN=ghp_xxx' vào file .env\n")
        sys.exit(1)

    try:
        run_push(
            backlog=backlog,
            repo=repo,
            phase_filter=args.phase,
            user_map=user_map,
            token=token,
        )
    except (RuntimeError, urllib.error.URLError, OSError) as e:
        print(f"\n❌ Lỗi trong quá trình push: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
