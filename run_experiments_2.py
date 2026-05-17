import subprocess
import time
import re
import csv
from pathlib import Path


OUTPUT_FILE = "experiment_2_results.csv"
LOG_DIR = Path("experiment_2_logs")

MATCHUPS = [
    {
        "name": "agent_vs_agentMMe1",
        "command": ["python", "-m", "referee", "agent", "agentMMe1"],
        "red_agent": "agent",
        "blue_agent": "agentMMe1",
        "runs": 10,
    },
    {
        "name": "agentMCTS_vs_agentMMe1",
        "command": ["python", "-m", "referee", "agentMCTS", "agentMMe1"],
        "red_agent": "agentMCTS",
        "blue_agent": "agentMMe1",
        "runs": 10,
    },
    {
        "name": "agentMCTS_vs_agent",
        "command": ["python", "-m", "referee", "agentMCTS", "agent"],
        "red_agent": "agentMCTS",
        "blue_agent": "agent",
        "runs": 5,
    },
]


def extract_turn_count(output: str):
    pattern = re.compile(
        r"(RED|BLUE)\s+to play\s+\(turn\s+(\d+)\).*?"
        r"\1\s+plays action",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(output)

    if matches:
        return int(matches[-1][1])

    return None


def extract_winner(output: str, red_agent: str, blue_agent: str):
    result_match = re.search(
        r"result:\s*player\s*(1|2)\s*\[([^:\]]+)",
        output,
        re.IGNORECASE
    )

    if result_match:
        player_num = result_match.group(1)

        if player_num == "1":
            return red_agent
        if player_num == "2":
            return blue_agent

    color_match = re.search(
        r"winner is\s+(red|blue)",
        output,
        re.IGNORECASE
    )

    if color_match:
        color = color_match.group(1).upper()

        if color == "RED":
            return red_agent
        if color == "BLUE":
            return blue_agent

    return "Unknown"


def run_one_match(matchup, run_id):
    start = time.perf_counter()

    result = subprocess.run(
        matchup["command"],
        capture_output=True,
        text=True
    )

    elapsed = time.perf_counter() - start
    output = result.stdout + "\n" + result.stderr

    winner = extract_winner(
        output,
        matchup["red_agent"],
        matchup["blue_agent"]
    )

    turn_count = extract_turn_count(output)

    log_file = LOG_DIR / f"{matchup['name']}_run_{run_id}.txt"
    log_file.write_text(output, encoding="utf-8")

    return {
        "Matchup": matchup["name"],
        "Run": run_id,
        "Red agent": matchup["red_agent"],
        "Blue agent": matchup["blue_agent"],
        "Winner": winner,
        "Turn count": turn_count,
        "Time taken seconds": round(elapsed, 4),
        "Return code": result.returncode,
        "Log file": str(log_file),
    }


def save_results_csv(rows):
    fieldnames = [
        "Matchup",
        "Run",
        "Red agent",
        "Blue agent",
        "Winner",
        "Turn count",
        "Time taken seconds",
        "Return code",
        "Log file",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    LOG_DIR.mkdir(exist_ok=True)

    rows = []

    for matchup in MATCHUPS:
        for run_id in range(1, matchup["runs"] + 1):
            print(f"Running {matchup['name']} | Run {run_id}/{matchup['runs']}")

            row = run_one_match(matchup, run_id)
            rows.append(row)

            print(
                f"  Winner: {row['Winner']} | "
                f"Turn: {row['Turn count']} | "
                f"Time: {row['Time taken seconds']}s"
            )

    save_results_csv(rows)

    print()
    print("Done.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Logs saved to: {LOG_DIR}/")


if __name__ == "__main__":
    main()