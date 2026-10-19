import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interview Agent Evaluation Runner")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ tập eval")
    parser.add_argument("--uc", nargs="+", help="Chạy eval theo danh sách UC cụ thể")
    args = parser.parse_args()

    report = {
        "status": "ready",
        "mode": "all" if args.all else f"uc_{args.uc}",
        "knowledge_dir_exists": Path("data/knowledge").exists(),
        "eval_data_dir_exists": Path("evals/data").exists(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
