import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_winner(output_text: str) -> str:
    lines = output_text.splitlines()

    for line in lines:
        lower_line = line.lower()
        if "winner is red" in lower_line:
            return "RED"
        if "winner is blue" in lower_line:
            return "BLUE"

    for line in lines:
        lower_line = line.lower()
        if "@ result: draw" in lower_line:
            return "DRAW"
        if "@ result: player 1" in lower_line:
            return "RED"
        if "@ result: player 2" in lower_line:
            return "BLUE"

    for line in lines:
        lower_line = line.lower()
        if "result" not in lower_line:
            continue
        if "draw" in lower_line:
            return "DRAW"

        match = re.search(r"player\s+([12])", lower_line)
        if match is None:
            continue
        if match.group(1) == "1":
            return "RED"
        if match.group(1) == "2":
            return "BLUE"

    return "UNKNOWN"


def parse_total_turns(output_text: str) -> int | None:
    matches = re.findall(r"to play \(turn\s+(\d+)\)", output_text, flags=re.IGNORECASE)
    if len(matches) > 0:
        values: list[int] = []
        for value in matches:
            values.append(int(value))
        return max(values)

    generic_matches = re.findall(r"\bturn\s+(\d+)\b", output_text, flags=re.IGNORECASE)
    if len(generic_matches) == 0:
        return None

    generic_values: list[int] = []
    for value in generic_matches:
        generic_values.append(int(value))
    return max(generic_values)


def run_one_game(
    project_root: Path,
    agent_name: str,
    red_weights: str,
    blue_weights: str,
    verbose_level: str,
    logs_dir: Path,
    game_index: int,
) -> dict:
    env = os.environ.copy()
    env["AGENTMCTS_WEIGHTS_RED"] = red_weights
    env["AGENTMCTS_WEIGHTS_BLUE"] = blue_weights

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
    total_turns = parse_total_turns(full_output)

    raw_output_path = logs_dir / f"batch_game_output_{game_index:04d}.txt"
    with raw_output_path.open("w", encoding="utf-8") as file:
        file.write(full_output)

    return {
        "game_index": game_index,
        "return_code": result.returncode,
        "winner": winner,
        "total_turns": total_turns,
        "red_weights": red_weights,
        "blue_weights": blue_weights,
        "raw_output_file": str(raw_output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="agentMCTS", help="Agent module name for both sides.")
    parser.add_argument("--games-per-side", type=int, default=10, help="Games for each side assignment.")
    parser.add_argument("--weights-a", required=True, help="Weights A: f1,f2,f3,f4,f5,f6,f7")
    parser.add_argument("--weights-b", required=True, help="Weights B: f1,f2,f3,f4,f5,f6,f7")
    parser.add_argument("--verbose", default="0", help="Referee verbose level.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    game_details: list[dict] = []
    game_index = 1

    for _ in range(args.games_per_side):
        game_result = run_one_game(
            project_root,
            args.agent,
            args.weights_a,
            args.weights_b,
            args.verbose,
            logs_dir,
            game_index,
        )
        game_result["side_setup"] = "A_as_RED_B_as_BLUE"
        game_details.append(game_result)
        game_index += 1

    for _ in range(args.games_per_side):
        game_result = run_one_game(
            project_root,
            args.agent,
            args.weights_b,
            args.weights_a,
            args.verbose,
            logs_dir,
            game_index,
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
    a_wins = 0
    b_wins = 0

    known_turn_counts: list[int] = []

    for game_result in game_details:
        if game_result["return_code"] != 0:
            failed_runs += 1

        winner = game_result["winner"]
        side_setup = game_result["side_setup"]

        if winner == "RED":
            red_wins += 1
            if side_setup == "A_as_RED_B_as_BLUE":
                a_wins += 1
            else:
                b_wins += 1
        elif winner == "BLUE":
            blue_wins += 1
            if side_setup == "A_as_RED_B_as_BLUE":
                b_wins += 1
            else:
                a_wins += 1
        elif winner == "DRAW":
            draws += 1
        else:
            unknown_results += 1

        turns = game_result["total_turns"]
        if turns is not None:
            known_turn_counts.append(turns)

    if total_games > 0:
        red_win_rate = red_wins / total_games
        blue_win_rate = blue_wins / total_games
        draw_rate = draws / total_games
        a_win_rate = a_wins / total_games
        b_win_rate = b_wins / total_games
        a_relative_strength = (a_wins + 0.5 * draws) / total_games
    else:
        red_win_rate = 0.0
        blue_win_rate = 0.0
        draw_rate = 0.0
        a_win_rate = 0.0
        b_win_rate = 0.0
        a_relative_strength = 0.0

    if len(known_turn_counts) > 0:
        average_total_turns = sum(known_turn_counts) / len(known_turn_counts)
    else:
        average_total_turns = None

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
        "a_wins": a_wins,
        "b_wins": b_wins,
        "a_win_rate": a_win_rate,
        "b_win_rate": b_win_rate,
        "a_relative_strength": a_relative_strength,
        "average_total_turns": average_total_turns,
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
    print(f"A relative strength: {a_relative_strength:.4f}")
    print(f"Average total turns: {average_total_turns}")


if __name__ == "__main__":
    main()

