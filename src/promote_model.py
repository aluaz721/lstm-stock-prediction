"""
Promotes a registered model version to the 'production' alias -- a
deliberate, separate step from training (see train.py's --promote flag
for the alternative of doing this immediately after training instead).

Usage:
    python -m src.promote_model --model-name stock-lstm --version 3
"""
import argparse

from src.registry import promote_to_production, get_production_version


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--version", type=int, required=True)
    args = p.parse_args()

    previous = get_production_version(args.model_name)
    promote_to_production(args.model_name, args.version)
    print(f"{args.model_name}: promoted v{args.version} to production (was v{previous})")


if __name__ == "__main__":
    main()
