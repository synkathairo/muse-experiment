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


def fetch_sgf_text(gid, delay):
    """Fetch one SGF, honoring 429s. Returns text or None."""
    url = f"{API}/api/v1/games/{gid}/sgf"
    while True:
        time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "60"))
                print(f"  429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            print(f"  game {gid}: {e}", flush=True)
            return None


def phase2_download(game_ids, sgf_dir, delay, workers):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    todo = [gid for gid in game_ids
            if not os.path.exists(os.path.join(sgf_dir, f"{gid}.sgf"))]
    print(f"downloading {len(todo)} SGFs ({workers} workers)...", flush=True)
    done = [0]
    lock = threading.Lock()

    def one(gid):
        text = fetch_sgf_text(gid, delay)
        ok = bool(text and "GM[1]" in text)
        if ok:
            with open(os.path.join(sgf_dir, f"{gid}.sgf"), "w") as f:
                f.write(text)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  {done[0]}/{len(todo)} SGFs", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, todo))
    n = sum(results)
    print(f"phase 2 done: {n} SGFs in {sgf_dir}", flush=True)
    return n


def is_bot(player_obj):
    ui = (player_obj or {}).get("ui_class", "")
    name = (player_obj or {}).get("username", "")
    return "bot" in ui.split() or "bot" in name.lower()


def phase1_discovery(args):
    """Crawl players -> ranked finished 9x9 games; returns list of game ids."""
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
            n_this_player = 0
            url = (f"{API}/api/v1/players/{pid}/games/?format=json"
                   f"&page_size=100&width=9&height=9&ordering=-ended")
            while url and len(seen_games) < args.max_games \
                    and n_this_player < args.max_games_per_player:
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
                    n_this_player += 1
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
    return list(seen_games)


def run(args):
    sgf_dir = os.path.join(args.out, "sgf")
    os.makedirs(sgf_dir, exist_ok=True)
    if args.phase2_only:
        # resume: read game ids straight from the existing manifest
        man = os.path.join(args.out, "manifest.jsonl")
        seen_games = []
        with open(man) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_games.append(json.loads(line)["id"])
        print(f"phase2-only: {len(seen_games)} games in manifest", flush=True)
    else:
        seen_games = phase1_discovery(args)

    # phase 2: download SGFs
    phase2_download(seen_games, sgf_dir, args.delay, args.workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=80000)
    ap.add_argument("--max-players", type=int, default=1500)
    ap.add_argument("--delay", type=float, default=0.7,
                    help="seconds between API requests")
    ap.add_argument("--seed-ladder-size", type=int, default=400,
                    help="use top-N ladder members as seeds")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel SGF download workers")
    ap.add_argument("--max-games-per-player", type=int, default=500,
                    help="cap games taken from a single player (forces breadth)")
    ap.add_argument("--phase2-only", action="store_true",
                    help="skip discovery; download SGFs for manifest.jsonl only")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
