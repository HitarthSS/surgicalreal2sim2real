"""Command-line entry points for preparation, execution and export."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stereo surgical-scene reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a new run without inference")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("config", type=Path)
    prepare.add_argument("output", type=Path)
    doctor = commands.add_parser("doctor", help="Check configured external assets and hashes")
    doctor.add_argument("config", type=Path)
    background_case = commands.add_parser("background-case", help="Build one background from an existing scene without foreground inference")
    background_case.add_argument("case", type=Path)
    background_case.add_argument("output", type=Path)
    background_case.add_argument("--config", type=Path)
    background_case.add_argument("--vlm", action="store_true", help="Read a credential from stdin for one bounded VLM selection")
    for name, help_text in {
        "run": "Execute a prepared run inside Slurm/Apptainer",
        "submit": "Submit one prepared run to Slurm",
        "credential": "Supply a hidden credential to a server RAM relay",
        "review": "Render heuristic and agent stereo overlays",
        "collect": "Collect verified geometry, joint angles and OBJ files",
        "background": "Add optional backgrounds to existing scenes without rerunning foreground stages",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "background-case":
        import json, sys
        from .background_stage import run_case
        cfg = json.loads(args.config.read_text()) if args.config else {}
        if args.vlm: cfg['vlm'] = True
        return run_case(args.case, args.output, cfg, sys.stdin.readline().strip() if cfg.get('vlm') else None)
    if args.command == "prepare":
        from .prepare import main as prepare_run
        return prepare_run(args.manifest, args.config, args.output)
    if args.command == "doctor":
        from .prepare import doctor as check_assets
        return check_assets(args.config)
    from importlib import import_module
    module = {
        "run": "runner", "submit": "submit", "credential": "credential_relay",
        "review": "review", "collect": "collect",
        "background": "background_stage",
    }[args.command]
    return import_module(f".{module}", __package__).main(args.root)


if __name__ == "__main__":
    main()
