"""SGF parsing for 9x9 game extraction.

Handles SGF collections, variations (main line only), AB/AW setup stones,
passes, and KGS quirks. Filtered to 9x9; everything else is dropped.
"""

import os
import zipfile
from dataclasses import dataclass, field


@dataclass
class Game:
    size: int = 19
    moves: list = field(default_factory=list)  # [('B'|'W', (r,c) | None)]
    setup_black: list = field(default_factory=list)
    setup_white: list = field(default_factory=list)
    result: str = ""
    handicap: int = 0
    black_name: str = ""
    white_name: str = ""
    to_play: str = "B"  # who moves first after setup (PL[], else handicap convention)


def sgf_point(s, size):
    """'aa' -> (0,0); row 0 = top, col 0 = left. '' or out-of-range -> None (pass)."""
    if len(s) < 2:
        return None
    c = ord(s[0]) - ord("a")
    r = ord(s[1]) - ord("a")
    if 0 <= r < size and 0 <= c < size:
        return (r, c)
    return None  # e.g. 'tt' on 9x9, or ''


def _unquote(v):
    """OGS sometimes wraps whole values in single quotes: RE['B+2.5']."""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


class _Parser:
    def __init__(self, text):
        self.s = text
        self.i = 0
        self.n = len(text)

    def peek(self):
        self._ws()
        return self.s[self.i] if self.i < self.n else ""

    def _ws(self):
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def _expect(self, ch):
        assert self.s[self.i] == ch, f"expected {ch!r} at {self.i}"
        self.i += 1

    def parse_collection(self):
        trees = []
        while self.peek() == "(":
            self.i += 1
            seq, _ = self._parse_tree()
            trees.append(seq)
        return trees

    def _parse_tree(self):
        """Parse one game tree; returns (main-line nodes, is_unbranched_chain).

        OGS writes the main line as nested single-node "variations"
        (;W[gd](;B[fc](;W[gc]...))); a variation that is itself an unbranched
        chain is spliced into the main line, real branches are ignored.
        """
        seq = []
        variations = []
        while True:
            c = self.peek()
            if c == ";":
                self.i += 1
                seq.append(self._parse_node())
            elif c == "(":
                self.i += 1
                variations.append(self._parse_tree())
            elif c == ")":
                self.i += 1
                break
            elif c == "":
                break
            else:
                self.i += 1  # tolerate stray chars
        if len(variations) == 1:
            sub_seq, sub_chain = variations[0]
            if sub_chain:
                seq.extend(sub_seq)
                return seq, True
        return seq, not variations

    def _parse_node(self):
        props = {}
        while True:
            self._ws()
            if self.i < self.n and self.s[self.i].isupper():
                start = self.i
                while self.i < self.n and self.s[self.i].isupper():
                    self.i += 1
                ident = self.s[start:self.i]
                vals = []
                while True:
                    self._ws()
                    if self.i < self.n and self.s[self.i] == "[":
                        self.i += 1
                        buf = []
                        while self.i < self.n:
                            ch = self.s[self.i]
                            if ch == "\\":
                                if self.i + 1 < self.n:
                                    buf.append(self.s[self.i + 1])
                                self.i += 2
                            elif ch == "]":
                                self.i += 1
                                break
                            else:
                                buf.append(ch)
                                self.i += 1
                        vals.append(_unquote("".join(buf)))
                    else:
                        break
                props[ident] = vals
            else:
                break
        return props


def parse_sgf(text):
    """Parse SGF text; return the first game (main line) as a Game, or None."""
    try:
        trees = _Parser(text).parse_collection()
    except (AssertionError, RecursionError, IndexError):
        return None
    if not trees or not trees[0]:
        return None
    seq = trees[0]
    root = seq[0]
    game = Game()
    try:
        game.size = int(root.get("SZ", ["19"])[0])
    except ValueError:
        return None
    if game.size != 9:
        return None
    game.result = root.get("RE", [""])[0]
    try:
        game.handicap = int(root.get("HA", ["0"])[0])
    except ValueError:
        game.handicap = 0
    game.black_name = root.get("PB", [""])[0]
    game.white_name = root.get("PW", [""])[0]
    for v in root.get("AB", []):
        p = sgf_point(v, 9)
        if p:
            game.setup_black.append(p)
    for v in root.get("AW", []):
        p = sgf_point(v, 9)
        if p:
            game.setup_white.append(p)
    pl = (root.get("PL", [""])[0] or "").strip().upper()
    if pl in ("B", "W"):
        game.to_play = pl
    elif game.setup_black and not game.setup_white:
        game.to_play = "W"  # handicap: black placed stones, white moves first
    for node in seq[1:]:
        for color in ("B", "W"):
            if color in node:
                vals = node[color]
                pt = sgf_point(vals[0] if vals else "", 9)
                game.moves.append((color, pt))
                break
    return game


def result_to_z(result):
    """'B+3.5'/'B+R' -> +1 (black win), 'W+...' -> -1, else None."""
    r = result.strip().upper()
    if r.startswith("B+"):
        return 1.0
    if r.startswith("W+"):
        return -1.0
    return None


def is_timeout(result):
    return "TIME" in result.strip().upper()


def looks_like_bot(name):
    return "bot" in name.lower()


def iter_sgf_texts(paths):
    """Yield (source_label, sgf_text) for .sgf files, dirs, and .zip archives."""
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    if f.endswith(".sgf"):
                        yield from iter_sgf_texts([os.path.join(root, f)])
        elif path.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".sgf"):
                        try:
                            raw = zf.read(name)
                        except Exception:
                            continue
                        yield f"{path}:{name}", raw.decode("utf-8", errors="replace")
        elif path.endswith(".sgf"):
            with open(path, "rb") as f:
                yield path, f.read().decode("utf-8", errors="replace")


def iter_games(paths):
    """Yield 9x9 Game objects from the given paths."""
    for label, text in iter_sgf_texts(paths):
        game = parse_sgf(text)
        if game is not None:
            yield game
