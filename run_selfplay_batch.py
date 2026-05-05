import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_winner(output_text: str) -> str:
    """
    Read referee output and return RED / BLUE / DRAW / UNKNOWN.
    """
    lines = output_text.splitlines()

    # Pattern 1: "... winner is RED/BLUE"
    for line in lines:
        lower_line = line.lower()
        if "winner is red" in lower_line:
            return "RED"
        if "winner is blue" in lower_line:
            return "BLUE"

    # Pattern 2: "... @ result: player 1/2 ..."
    for line in lines:
        lower_line = line.lower()
        if "@ result: draw" in lower_line:
            return "DRAW"
        if "@ result: player 1" in lower_line:
            return "RED"
        if "@ result: player 2" in lower_line:
            return "BLUE"

    # Pattern 3: any line that contains "result" and "player <number>".
    for line in lines:
        lower_line = line.lower()
        if "result" not in lower_line:
            continue

        if "draw" in lower_line:
            return "DRAW"

        match = re.search(r"player\s+([12])", lower_line)
        if match is None:
            continue

        player_number = match.group(1)
        if player_number == "1":
            return "RED"
        if player_number == "2":
            return "BLUE"

    return "UNKNOWN"


def run_one_game(
    project_root: Path,
    agent_name: str,
    red_weights: str,
    blue_weights: str,
    verbose_level: str,
    logs_dir: Path,
    game_index: int
) -> dict:
    env = os.environ.copy()
    env["AGENTMM_WEIGHTS_RED"] = red_weights
    env["AGENTMM_WEIGHTS_BLUE"] = blue_weights

    command = [
        sys.executable,
        "-m",
        "referee",
        "-v",
        verbose_level,
        agent_name,
        agent_name,
    ]

    result = subprocess.run(
        command,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    full_output = result.stdout + "\n" + result.stderr
    winner = parse_winner(full_output)

    # Save raw output for every game so we can debug parsing issues.
    raw_output_path = logs_dir / f"batch_game_output_{game_index:04d}.txt"
    with raw_output_path.open("w", encoding="utf-8") as file:
        file.write(full_output)

    # If winner is still unknown, save a highlighted copy for quick inspection.
    unknown_output_path = None
    if winner == "UNKNOWN":
        unknown_output_path = logs_dir / f"unknown_game_{game_index:04d}.txt"
        with unknown_output_path.open("w", encoding="utf-8") as file:
            file.write(full_output)

    game_result = {
        "game_index": game_index,
        "return_code": result.returncode,
        "winner": winner,
        "red_weights": red_weights,
        "blue_weights": blue_weights,
        "raw_output_file": str(raw_output_path),
        "unknown_output_file": str(unknown_output_path) if unknown_output_path is not None else None,
    }
    return game_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="agentMM", help="Agent module name for both sides.")
    parser.add_argument("--games-per-side", type=int, default=10, help="Games for each side assignment.")
    parser.add_argument("--weights-a", required=True, help="Weights A: f1,f2,f3,f4,f5,f6")
    parser.add_argument("--weights-b", required=True, help="Weights B: f1,f2,f3,f4,f5,f6")
    parser.add_argument("--verbose", default="0", help="Referee verbose level.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    game_details: list[dict] = []

    game_index = 1

    for _ in range(args.games_per_side):
        game_result = run_one_game(
            project_root=project_root,
            agent_name=args.agent,
            red_weights=args.weights_a,
            blue_weights=args.weights_b,
            verbose_level=args.verbose,
            logs_dir=logs_dir,
            game_index=game_index,
        )
        game_result["side_setup"] = "A_as_RED_B_as_BLUE"
        game_details.append(game_result)
        game_index += 1

    for _ in range(args.games_per_side):
        game_result = run_one_game(
            project_root=project_root,
            agent_name=args.agent,
            red_weights=args.weights_b,
            blue_weights=args.weights_a,
            verbose_level=args.verbose,
            logs_dir=logs_dir,
            game_index=game_index,
        )
        game_result["side_setup"] = "B_as_RED_A_as_BLUE"
        game_details.append(game_result)
        game_index += 1

    total_games = len(game_details)
    red_wins = 0
    blue_wins = 0
    draws = 0
    unknown_results = 0
    failed_runs = 0

    for game_result in game_details:
        if game_result["return_code"] != 0:
            failed_runs += 1

        if game_result["winner"] == "RED":
            red_wins += 1
        elif game_result["winner"] == "BLUE":
            blue_wins += 1
        elif game_result["winner"] == "DRAW":
            draws += 1
        else:
            unknown_results += 1

    if total_games > 0:
        red_win_rate = red_wins / total_games
        blue_win_rate = blue_wins / total_games
        draw_rate = draws / total_games
    else:
        red_win_rate = 0.0
        blue_win_rate = 0.0
        draw_rate = 0.0

    summary = {
        "agent": args.agent,
        "games_per_side": args.games_per_side,
        "total_games": total_games,
        "weights_a": args.weights_a,
        "weights_b": args.weights_b,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "unknown_results": unknown_results,
        "failed_runs": failed_runs,
        "red_win_rate": red_win_rate,
        "blue_win_rate": blue_win_rate,
        "draw_rate": draw_rate,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = logs_dir / f"batch_summary_{timestamp}.json"
    detail_path = logs_dir / f"batch_detail_{timestamp}.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with detail_path.open("w", encoding="utf-8") as file:
        json.dump(game_details, file, indent=2, ensure_ascii=False)

    print("Batch run finished.")
    print(f"Summary file: {summary_path}")
    print(f"Detail file: {detail_path}")
    print(f"Total games: {total_games}")
    print(f"RED wins: {red_wins} ({red_win_rate:.2%})")
    print(f"BLUE wins: {blue_wins} ({blue_win_rate:.2%})")
    print(f"DRAWs: {draws} ({draw_rate:.2%})")
    print(f"Unknown results: {unknown_results}")
    print(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    main()
