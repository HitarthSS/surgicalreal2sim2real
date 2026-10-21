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
    for name, help_text in {
        "run": "Execute a prepared run inside Slurm/Apptainer",
        "submit": "Submit one prepared run to Slurm",
        "credential": "Supply a hidden credential to a server RAM relay",
        "review": "Render heuristic and agent stereo overlays",
        "collect": "Collect verified geometry, joint angles and OBJ files",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("root", type=Path)
    args = parser.parse_args(argv)
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
    }[args.command]
    return import_module(f".{module}", __package__).main(args.root)


if __name__ == "__main__":
    main()
