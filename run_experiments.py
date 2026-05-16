import subprocess
import time
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

#import matplotlib.pyplot as plt


OUTPUT_FILE = "experiment_results.csv"
SUMMARY_FILE = "experiment_summary.csv"

LOG_DIR = Path("experiment_logs")
GRAPH_DIR = Path("experiment_graphs")

MATCHUPS = [
    {
        "name": "agentMMe1_vs_agent",
        "command": ["python", "-m", "referee", "agentMMe1", "agent"],
        "red_agent": "agentMMe1",
        "blue_agent": "agent",
        "runs": 10,
    },
    {
        "name": "agentMMe1_vs_agentMCTS",
        "command": ["python", "-m", "referee", "agentMMe1", "agentMCTS"],
        "red_agent": "agentMMe1",
        "blue_agent": "agentMCTS",
        "runs": 10,
    },
    {
        "name": "agent_vs_agentMCTS",
        "command": ["python", "-m", "referee", "agent", "agentMCTS"],
        "red_agent": "agent",
        "blue_agent": "agentMCTS",
        "runs": 5,
    },
]


def extract_turn_count(output: str):
    """
    Extract the turn count of the last played action.

    Example:
    BLUE to play (turn 26) ...
    BLUE plays action EAT(...)
    game over

    Final completed turn = 26
    """
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


def build_summary(rows):
    summary_rows = []

    for matchup in sorted(set(row["Matchup"] for row in rows)):
        data = [row for row in rows if row["Matchup"] == matchup]

        win_counts = Counter(row["Winner"] for row in data)
        total_games = len(data)

        valid_times = [row["Time taken seconds"] for row in data]
        valid_turns = [
            row["Turn count"]
            for row in data
            if row["Turn count"] is not None
        ]

        avg_time = sum(valid_times) / len(valid_times) if valid_times else None
        avg_turn = sum(valid_turns) / len(valid_turns) if valid_turns else None

        agents = sorted(set([row["Red agent"] for row in data] + [row["Blue agent"] for row in data]))

        for agent in agents:
            wins = win_counts.get(agent, 0)
            win_rate = wins / total_games if total_games > 0 else 0

            summary_rows.append({
                "Matchup": matchup,
                "Agent": agent,
                "Games": total_games,
                "Wins": wins,
                "Win rate": round(win_rate, 4),
                "Win rate percent": round(win_rate * 100, 2),
                "Average time seconds": round(avg_time, 4) if avg_time is not None else None,
                "Average turn count": round(avg_turn, 2) if avg_turn is not None else None,
            })

    return summary_rows


def save_summary_csv(summary_rows):
    fieldnames = [
        "Matchup",
        "Agent",
        "Games",
        "Wins",
        "Win rate",
        "Win rate percent",
        "Average time seconds",
        "Average turn count",
    ]

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(exist_ok=True)

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

    summary_rows = build_summary(rows)
    save_summary_csv(summary_rows)

    #plot_runtime_line(rows)
    #plot_runtime_dot_by_winner(rows)
    #plot_win_efficiency(summary_rows)
    #plot_turn_count_line(rows)

    print()
    print("Done.")
    print(f"Raw results saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Logs saved to: {LOG_DIR}/")
    print(f"Graphs saved to: {GRAPH_DIR}/")


if __name__ == "__main__":
    main()