"""Polite bulk pull of 9x9 ranked OGS games -> data SGFs.

Strategy:
  1. seed players from the site 9x9 ladder (/api/v1/ladders/315/players/)
  2. for each player, paginate /players/{id}/games/?width=9&height=9
  3. keep ranked, non-bot games; snowball unseen opponents (bounded)
  4. download /games/{id}/sgf with a politeness delay; skip files already
     on disk so the pull is resumable

Politesse: ~1.4 req/s, honor 429 with Retry-After, identifiable User-Agent.
Per PLAN.md §4 / ToS review: official public API, cached locally, rate-limited.

Usage:
    python -m gotrain.ogs_pull --out data/ogs --max-games 3000   # validation
    python -m gotrain.ogs_pull --out data/ogs --max-games 80000   # full (background)

Output: <out>/sgf/{game_id}.sgf + <out>/manifest.jsonl (one JSON per game).
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

API = "https://online-go.com"
UA = "weiqi-demo-research/0.1 (single-machine dataset pull; contact via repo)"
LADDER_9X9 = 315


def api_get(url, delay):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    while True:
        time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "60"))
                print(f"  429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


def api_get_text(url, delay):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    while True:
        time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "60"))
                print(f"  429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


def is_bot(player_obj):
    ui = (player_obj or {}).get("ui_class", "")
    name = (player_obj or {}).get("username", "")
    return "bot" in ui.split() or "bot" in name.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=80000)
    ap.add_argument("--max-players", type=int, default=1500)
    ap.add_argument("--delay", type=float, default=0.7,
                    help="seconds between API requests")
    ap.add_argument("--seed-ladder-size", type=int, default=400,
                    help="use top-N ladder members as seeds")
    args = ap.parse_args()

    sgf_dir = os.path.join(args.out, "sgf")
    os.makedirs(sgf_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")

    seen_games = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                try:
                    seen_games.add(json.loads(line)["id"])
                except Exception:
                    pass
    manifest = open(manifest_path, "a")

    print("fetching 9x9 ladder members...", flush=True)
    seeds = []
    page = 1
    # NOTE: this endpoint caps page_size at 50 and omits the `next` URL,
    # so paginate manually with ?page=N.
    while len(seeds) < args.seed_ladder_size:
        d = api_get(f"{API}/api/v1/ladders/{LADDER_9X9}/players/?format=json"
                    f"&page_size=50&page={page}", args.delay)
        rows = d["results"]
        if not rows:
            break
        for row in rows:
            seeds.append(row["player"]["id"])
            if len(seeds) >= args.seed_ladder_size:
                break
        if len(rows) < 50:
            break
        page += 1
    print(f"{len(seeds)} seed players", flush=True)

    queue = list(seeds)
    seen_players = set(seeds)
    n_players = 0
    t0 = time.time()
    try:
        while queue and len(seen_games) < args.max_games and n_players < args.max_players:
            pid = queue.pop(0)
            n_players += 1
            url = (f"{API}/api/v1/players/{pid}/games/?format=json"
                   f"&page_size=100&width=9&height=9&ordering=-ended")
            while url and len(seen_games) < args.max_games:
                d = api_get(url, args.delay)
                for g in d["results"]:
                    gid = g["id"]
                    if gid in seen_games:
                        continue
                    if not g.get("ranked"):
                        continue
                    if not g.get("ended"):
                        continue  # in progress: SGF needs auth
                    if g.get("annulled"):
                        continue
                    if (g.get("outcome") or "") in ("Cancellation",):
                        continue
                    bp, wp = g["players"]["black"], g["players"]["white"]
                    if is_bot(bp) or is_bot(wp):
                        continue
                    seen_games.add(gid)
                    manifest.write(json.dumps({
                        "id": gid,
                        "black": g.get("black"), "white": g.get("white"),
                        "rules": g.get("rules"), "handicap": g.get("handicap"),
                        "outcome": g.get("outcome"), "ended": g.get("ended"),
                    }) + "\n")
                    for oid in (g.get("black"), g.get("white")):
                        if oid and oid not in seen_players and len(seen_players) < args.max_players * 4:
                            seen_players.add(oid)
                            queue.append(oid)
                    if len(seen_games) >= args.max_games:
                        break
                url = d.get("next")
            if n_players % 25 == 0:
                el = time.time() - t0
                print(f"  players={n_players} games={len(seen_games)} "
                      f"queue={len(queue)} elapsed={el/60:.1f}m", flush=True)
    finally:
        manifest.close()
    print(f"phase 1 done: {len(seen_games)} games from {n_players} players", flush=True)

    # phase 2: download SGFs
    todo = [gid for gid in seen_games
            if not os.path.exists(os.path.join(sgf_dir, f"{gid}.sgf"))]
    print(f"downloading {len(todo)} SGFs...", flush=True)
    done = 0
    for gid in todo:
        try:
            text = api_get_text(f"{API}/api/v1/games/{gid}/sgf", args.delay)
        except Exception as e:
            print(f"  game {gid}: {e}", flush=True)
            continue
        if "GM[1]" not in text:
            continue
        with open(os.path.join(sgf_dir, f"{gid}.sgf"), "w") as f:
            f.write(text)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(todo)} SGFs", flush=True)
    print(f"phase 2 done: {done} SGFs in {sgf_dir}", flush=True)


if __name__ == "__main__":
    main()
