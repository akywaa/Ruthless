#!/usr/bin/env python3
"""Minimal UCI match runner for Ruthless.

Plays a fixed-nodes (or fixed-movetime) match between two UCI engines,
alternating colours over a set of openings, and prints an SPRT summary.

Both engines must honour the chosen limit. Ruthless supports `go nodes`;
engines that only understand `go movetime` (for example Patricia 5.1) must be
tested with --movetime.

Examples:
    python tools/match.py --a target/release/Ruthless.exe \
                          --b target/release/Ruthless-prev.exe --games 400 \
                          --nodes 20000 --sprt 0 5
    python tools/match.py --a target/release/Ruthless.exe \
                          --b target/release/patricia_v3.exe --games 100 \
                          --movetime 100 --pgn games.pgn
"""

import argparse
import math
import multiprocessing as mp
import sys
import time

import chess
import chess.engine
import chess.pgn

OPENINGS = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"],
    ["g1f3", "g8f6", "c2c4", "e7e6", "b1c3", "d7d5"],
    ["e2e4", "e7e5", "f1c4", "g8f6", "d2d3", "f8c5"],
    ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5"],
]


def parse_engine(spec):
    """Split an ENGINE[:OPTION=VALUE,...] spec into path and options."""
    if ":" in spec and not spec.endswith(".exe"):
        path, _, opts = spec.partition(":")
        options = {}
        for item in opts.split(","):
            k, _, v = item.partition("=")
            if k:
                options[k.strip()] = parse_value(v.strip())
        return path, options
    return spec, {}


def parse_value(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def play_game(job):
    """Play a single game and return (result_for_a, pgn_text, error)."""
    idx, opening, a_spec, b_spec, limit, max_plies, _seed, pgn_notes = job
    a_path, a_opts = parse_engine(a_spec)
    b_path, b_opts = parse_engine(b_spec)

    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    game = chess.pgn.Game.from_board(board)

    white_is_a = idx % 2 == 0
    white_path, white_opts = (a_path, a_opts) if white_is_a else (b_path, b_opts)
    black_path, black_opts = (b_path, b_opts) if white_is_a else (a_path, a_opts)

    game.headers["Event"] = "sprt"
    game.headers["Round"] = str(idx)
    for note in pgn_notes:
        game.headers[note[0]] = note[1]

    error = None
    try:
        white = chess.engine.SimpleEngine.popen_uci(white_path)
        black = chess.engine.SimpleEngine.popen_uci(black_path)
    except Exception as exc:  # engine failed to start
        return 0.0, None, f"startup failed: {exc}"

    try:
        for opts, engine in ((white_opts, white), (black_opts, black)):
            configure(engine, opts)
        game.headers["White"] = white.id.get("name", white_path)
        game.headers["Black"] = black.id.get("name", black_path)

        node = game.end()
        while board.outcome(claim_draw=True) is None and len(board.move_stack) < max_plies:
            engine = white if board.turn == chess.WHITE else black
            try:
                result = engine.play(board, limit)
            except chess.engine.EngineTerminatedError as exc:
                error = f"engine died: {exc}"
                break
            if result.move is None or result.move not in board.legal_moves:
                error = f"illegal/no move: {result.move}"
                break
            board.push(result.move)
            node = node.add_variation(result.move)

        outcome = board.outcome(claim_draw=True)
        game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"

        if error is not None:
            # A crashed engine loses; a crash before any move is a draw.
            score = 0.0 if white_is_a else 1.0
            game.headers["Result"] = "0-1" if white_is_a else "1-0"
        elif outcome is None:
            score = 0.5
        else:
            white_score = 1.0 if outcome.winner == chess.WHITE else (
                0.0 if outcome.winner == chess.BLACK else 0.5
            )
            score = white_score if white_is_a else 1.0 - white_score
    finally:
        for engine in (white, black):
            try:
                engine.quit()
            except Exception:
                pass

    game.headers["Termination"] = error if error else (
        str(board.outcome(claim_draw=True).termination.name) if board.outcome(claim_draw=True) else "maxplies"
    )
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return score, game.accept(exporter), error


def configure(engine, options):
    defaults = {"Threads": 1, "Hash": 64}
    defaults.update(options)
    supported = engine.options
    wanted = {k: v for k, v in defaults.items() if k in supported}
    if wanted:
        engine.configure(wanted)


def elo_diff(score, n):
    score = min(max(score, 1e-9), 1 - 1e-9)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    sd = math.sqrt(score * (1.0 - score) / max(n, 1))
    # Spread of the score estimate around the observed score.
    elo_err = 400.0 / math.log(10.0) * sd / (score * (1.0 - score))
    return elo, elo_err


def los(elo, elo_err):
    if elo_err <= 0:
        return 1.0 if elo > 0 else 0.0
    return 0.5 * (1.0 + math.erf(elo / (elo_err * math.sqrt(2.0))))


def llr(wins, draws, losses, elo0, elo1):
    """Logistic SPRT log-likelihood ratio (draws count as half a win)."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    score = min(max(score, 1e-4), 1 - 1e-4)

    def expected(p):
        return 1.0 / (1.0 + 10.0 ** (-p / 400.0))

    s0, s1 = expected(elo0), expected(elo1)
    return (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * s0 * (1.0 - s0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="engine A (path[:opt=val,...])")
    ap.add_argument("--b", required=True, help="engine B (path[:opt=val,...])")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--nodes", type=int, default=20000, help="nodes per move")
    ap.add_argument("--movetime", type=int, default=0, help="ms per move instead of --nodes")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    ap.add_argument("--pgn", help="write games to this PGN file")
    args = ap.parse_args()

    if args.movetime > 0:
        limit = chess.engine.Limit(time=args.movetime / 1000.0)
    else:
        limit = chess.engine.Limit(nodes=args.nodes)

    jobs = []
    for i in range(args.games):
        # Each opening is played twice, once with each colour assignment.
        opening = OPENINGS[(i // 2) % len(OPENINGS)]
        notes = [("Nodes", str(args.nodes))] if args.movetime == 0 else []
        jobs.append((
            i,
            list(opening),
            args.a,
            args.b,
            limit,
            args.max_plies,
            args.seed + i,
            notes,
        ))

    wins = draws = losses = 0
    errors = 0
    pgn_chunks = []
    start = time.time()

    with mp.Pool(args.concurrency) as pool:
        for score, pgn, error in pool.imap_unordered(play_game, jobs):
            if error:
                errors += 1
            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1
            if pgn:
                pgn_chunks.append(pgn)
            played = wins + draws + losses
            if played % 10 == 0 or played == args.games:
                n = played
                score_pct = (wins + 0.5 * draws) / n
                elo, err = elo_diff(score_pct, n)
                print(
                    f"games {n}/{args.games}  +{wins} ={draws} -{losses}  "
                    f"score {100 * score_pct:.1f}%  elo {elo:+.0f} +/- {err:.0f}  "
                    f"LOS {100 * los(elo, err):.1f}%  ({time.time() - start:.0f}s)",
                    flush=True,
                )

    n = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / max(n, 1)
    elo, err = elo_diff(score_pct, n)
    print()
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"games {n}: +{wins} ={draws} -{losses} ({100 * score_pct:.1f}%)")
    print(f"elo {elo:+.0f} +/- {err:.0f}, LOS {100 * los(elo, err):.1f}%, errors {errors}")

    if args.sprt:
        value = llr(wins, draws, losses, args.sprt[0], args.sprt[1])
        bounds = math.log(1.0 / 0.05)
        print(f"llr {value:+.2f} (bounds +/-{bounds:.2f})")
        if value >= bounds:
            print("SPRT: accept H1")
        elif value <= -bounds:
            print("SPRT: accept H0")
        else:
            print("SPRT: continue")

    if args.pgn and pgn_chunks:
        with open(args.pgn, "w", encoding="utf-8") as handle:
            for chunk in pgn_chunks:
                handle.write(chunk)
                handle.write("\n\n")
        print(f"pgn written to {args.pgn}")


if __name__ == "__main__":
    sys.exit(main())
