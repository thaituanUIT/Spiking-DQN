import argparse
import importlib
import sys


COMMANDS = {
    ("baseline", "train"): ("baseline.train", "main"),
    ("baseline", "test"): ("baseline.test", "main"),
    ("baseline", "render"): ("baseline.render", "main"),
    ("v2", "train"): ("v2.train", "main"),
    ("v2", "test"): ("v2.test", "main"),
    ("v2", "render"): ("v2.render", "main"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for SpikingDQN baseline and v2 experiments."
    )
    parser.add_argument("stack", choices=["baseline", "v2"], help="Experiment stack")
    parser.add_argument("command", choices=["train", "test", "render"], help="Action")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()
    module_name, function_name = COMMANDS[(parsed.stack, parsed.command)]
    entrypoint = getattr(importlib.import_module(module_name), function_name)

    forwarded_args = parsed.args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    sys.argv = [f"{parsed.stack} {parsed.command}"] + forwarded_args
    entrypoint()


if __name__ == "__main__":
    main()
