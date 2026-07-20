"""
Calls the admin refresh-and-check-drift endpoint. Meant to run on a
schedule -- but NOT as a GitHub Actions cron job while this is deployed
locally: GitHub's hosted runners aren't on your network and structurally
cannot reach `localhost` on your machine. This is a plain script instead,
runnable via your OS's own scheduler:

    macOS/Linux (cron): add a line like this to `crontab -e`
        0 18 * * 1-5  cd /path/to/quantum-stock-mlops && ./venv/bin/python scripts/refresh_and_check_drift.py

    Or just run it manually whenever you want a fresh check:
        python scripts/refresh_and_check_drift.py

Once this is deployed somewhere with a real, internet-reachable URL
(Render, Fly, ECS -- see the deployment conversation), THIS script's
logic moves into an actual GitHub Actions scheduled workflow that curls
the deployed endpoint instead. Nothing about the admin endpoint itself
needs to change for that migration -- only where the trigger lives.

Usage:
    export ADMIN_API_TOKEN=local-dev-token   # must match the backend's env var
    python scripts/refresh_and_check_drift.py [--base-url http://localhost:8000]
"""
import argparse
import os
import sys

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    token = os.environ.get("ADMIN_API_TOKEN")
    if not token:
        print("ADMIN_API_TOKEN is not set -- export it before running this script.")
        sys.exit(1)

    resp = requests.post(
        f"{args.base_url}/admin/refresh-and-check-drift",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,  # fetching + feature computation for every ticker can take a while
    )

    if resp.status_code != 200:
        print(f"Drift check failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    body = resp.json()
    print(f"Drift check completed at {body['checked_at']}")

    any_drifted = False
    for ticker, features in body["results"].items():
        if "error" in features:
            print(f"  {ticker}: ERROR -- {features['error']}")
            continue
        drifted_features = [
            name for name, result in features.items()
            if result.get("is_drifted")
        ]
        if drifted_features:
            any_drifted = True
            print(f"  {ticker}: DRIFT DETECTED in {drifted_features}")
        else:
            checked = [n for n, r in features.items() if "psi_score" in r]
            print(f"  {ticker}: no drift ({len(checked)} features checked)")

    if any_drifted:
        print("\nAt least one ticker/feature shows drift -- consider retraining.")
        sys.exit(2)  # distinct exit code so a cron wrapper could alert on this specifically


if __name__ == "__main__":
    main()
