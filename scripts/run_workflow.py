"""Run existing rat-group workflow scripts from a single entry point."""

import argparse
import json
import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_PATH = PROJECT_ROOT / "configs" / "workflows.json"


def load_workflows():
    with WORKFLOWS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_workflows(workflows):
    for workflow, groups in workflows.items():
        print(workflow)
        for group, script in groups.items():
            print(f"  {group}: {script}")


def main():
    parser = argparse.ArgumentParser(
        description="Run an existing LFP workflow script by workflow and rat group."
    )
    parser.add_argument(
        "workflow", nargs="?", help="Workflow name, e.g. ripple_threshold"
    )
    parser.add_argument("group", nargs="?", help="Rat group, e.g. buildup_r1_4")
    parser.add_argument("--list", action="store_true", help="List available workflows")
    args = parser.parse_args()

    workflows = load_workflows()
    if args.list:
        list_workflows(workflows)
        return

    if not args.workflow or not args.group:
        parser.error("workflow and group are required unless --list is used")

    try:
        script = workflows[args.workflow][args.group]
    except KeyError as exc:
        raise SystemExit(f"Unknown workflow/group combination: {exc}") from exc

    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise SystemExit(f"Workflow script does not exist: {script_path}")

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
