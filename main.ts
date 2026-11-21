// Sokoban solver: per-room push-optimal A* over push states.
//
// Structure of a level (see the competition rules): a one-square-wide hallway
// holds the player, and 1-4 rooms open off it through one door each. A box
// pushed out of a room can never be pushed back, so every box stays in its
// room and the level's push-optimal count is the sum of its rooms' optima.
// This program therefore splits a level into rooms and solves each room
// independently with a push-optimal search, then concatenates the rooms'
// move strings. Solving a room is a search over push states: a state is the
// boxes' squares plus the player's reachable region (named by its lowest
// square). The search is A* ordered by pushes made plus an admissible
// heuristic (the cheapest assignment of boxes to goals by grid distance), so
// the first solved state popped is push-optimal. Two deadlocks are pruned:
// a box on a dead square (one from which no push, ignoring the other boxes,
// can ever reach a goal) and a frozen 2x2 block of walls and boxes holding a
// box off its goal. Neither prunes a solvable state.
//
// Deterministic by construction: the search is bounded by work (a cap on
// generated states), never by the clock; ties break by square order and by
// insertion order. No clock, no randomness, no environment reads beyond the
// input/output directory overrides the kit's starter uses.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work cap per room search: generated states. The generator only keeps a
 *  room its own push search solves within 250,000 states, and A* with an
 *  admissible heuristic expands a subset of that, so this cap is headroom. */
const MAX_STATES_PER_ROOM = 400_000;
/** Work cap for the whole-level fallback search. */
const MAX_STATES_WHOLE = 400_000;
/** Work cap for the move-space fallback (a plain A* over steps). */
const MAX_STATES_MOVES = 200_000;

type Dir = [number, number, string];
const DIRS: Dir[] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box can be pushed onto some goal, ignoring the other
 *  boxes: pull the goals backwards. A box on square y can be pushed to x
 *  (one step along a direction) when the player can stand at y - d. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Shortest grid distances from every goal over open floor (boxes ignored),
 *  one Int32Array per goal; -1 where unreachable. */
function goalDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[q] === -1) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

/** All orderings of the goal indices, for the assignment heuristic. */
function permutations(n: number): number[][] {
  if (n === 0) return [[]];
  const rest = permutations(n - 1);
  const out: number[][] = [];
  for (const r of rest) {
    for (let i = 0; i <= r.length; i++) {
      const p = r.slice();
      p.splice(i, 0, n - 1);
      out.push(p);
    }
  }
  return out;
}

/** Flood-fill the player's reachable region from `from` with `occ` in the
 *  way, marking `seen` with `stamp`; returns the region's lowest square. */
function reachLow(
  level: Level,
  occ: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && !occ[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
 *  and boxes holding a box off its goal. */
function frozenBlock(level: Level, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

interface PushSolution {
  pushes: number;
  moves: string;
}

/** Pack sorted box squares into one number (exact while size ** (n + 1) stays
 *  under 2 ** 53; rooms here hold at most 4 boxes on boards under 256
 *  squares). Falls back to -1 when the board is too large. */
function packBoxes(boxes: number[], base: number): number {
  let k = 0;
  for (const b of boxes) k = k * base + b;
  return k;
}

function unpackBoxes(pack: number, n: number, base: number, out: number[]): void {
  for (let i = n - 1; i >= 0; i--) {
    out[i] = pack % base;
    pack = Math.floor(pack / base);
  }
}

/**
 * Push-optimal solution of `level` by A* over push states, at most
 * `maxStates` generated states; {pushes, moves} (moves in lower-case LURD),
 * or null when the search gives up (or the level has no solution).
 */
function solvePushOptimal(level: Level, maxStates: number): PushSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  if (n === 0) return { pushes: 0, moves: '' };
  if (goals.length !== n) return null;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const dists = goalDistances(level);
  const perms = permutations(n);
  const heuristic = (boxes: number[]): number => {
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[p[i]!]![boxes[i]!]!;
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const base = size;
  const usePack = Math.pow(base, n + 1) < 2 ** 53;
  const keyOf = (boxes: number[], low: number): string | number => {
    if (usePack) {
      let k = low;
      for (const b of boxes) k = k * base + b;
      return k;
    }
    return `${low}:${boxes.join(',')}`;
  };

  // States, in discovery order: packed boxes, player square, parent, pushes
  // made, and the push that made the state (from square, to square).
  const stBoxes: number[] = [];
  const stBoxesList: number[][] | null = usePack ? null : [];
  const stPlayer: number[] = [];
  const stParent: number[] = [];
  const stG: number[] = [];
  const stFrom: number[] = [];
  const stTo: number[] = [];
  const getBoxes = (s: number, out: number[]): void => {
    if (usePack) unpackBoxes(stBoxes[s]!, n, base, out);
    else for (let i = 0; i < n; i++) out[i] = stBoxesList![s]![i]!;
  };

  for (const b of level.boxes) occ[b] = 1;
  const startLow = reachLow(level, occ, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occ[b] = 0;
  const startKey = keyOf(level.boxes, startLow);
  stBoxes.push(usePack ? packBoxes(level.boxes, base) : 0);
  if (!usePack) stBoxesList!.push(level.boxes.slice());
  stPlayer.push(level.player);
  stParent.push(-1);
  stG.push(0);
  stFrom.push(-1);
  stTo.push(-1);
  const best = new Map<string | number, number>();
  best.set(startKey, 0);

  // Binary heap of [f, -g, insertion sequence, state], smallest first.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const hLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!hLess(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && hLess(heap[c + 1]!, heap[c]!)) c++;
        if (!hLess(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  heapPush([heuristic(level.boxes), 0, seq++, 0]);

  const cur = new Array<number>(n);
  const next = new Array<number>(n);
  let found = -1;
  let expanded = 0;
  while (heap.length > 0) {
    const [, , , s] = heapPop();
    const g = stG[s]!;
    const sKey = (() => {
      getBoxes(s, cur);
      for (const b of cur) occ[b] = 1;
      const low = reachLow(level, occ, stPlayer[s]!, seen, ++stamp, stack);
      for (const b of cur) occ[b] = 0;
      return keyOf(cur, low);
    })();
    if (best.get(sKey) !== s) continue; // a better copy of this state exists
    getBoxes(s, cur);
    if (solved(cur)) {
      found = s;
      break;
    }
    expanded++;
    if (expanded > maxStates) return null;
    for (const b of cur) occ[b] = 1;
    reachLow(level, occ, stPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < n; bi++) {
      const b = cur[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const frozen = frozenBlock(level, occ, goal, t);
        if (!frozen) {
          for (let i = 0; i < n; i++) next[i] = cur[i]!;
          next[bi] = t;
          next.sort((a, b2) => a - b2);
        }
        occ[t] = 0;
        occ[b] = 1;
        if (frozen) continue;
        // The heuristic first: a box cut off from every goal by walls can
        // never reach one, so this state can never solve.
        const hh = heuristic(next);
        if (hh === Infinity) continue;
        // The new region, from the square the player stands on after the
        // push, with the new boxes in the way.
        occ[b] = 0;
        occ[t] = 1;
        const low = reachLow(level, occ, b, mark, ++stamp, stack);
        occ[t] = 0;
        occ[b] = 1;
        const key = keyOf(next, low);
        const prevIdx = best.get(key);
        // A state reached again with no fewer pushes adds nothing; reached
        // with fewer, it replaces the older copy (the stale copy is dropped
        // when popped, by the best check there).
        if (prevIdx !== undefined && stG[prevIdx]! <= g + 1) continue;
        if (stBoxes.length >= maxStates) {
          for (const x of cur) occ[x] = 0;
          return null;
        }
        const idx = stBoxes.length;
        best.set(key, idx);
        stBoxes.push(usePack ? packBoxes(next, base) : 0);
        if (!usePack) stBoxesList!.push(next.slice());
        stPlayer.push(b);
        stParent.push(s);
        stG.push(g + 1);
        stFrom.push(b);
        stTo.push(t);
        heapPush([g + 1 + hh, -(g + 1), seq++, idx]);
      }
    }
    for (const x of cur) occ[x] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = stParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const from = stFrom[s]!;
    const to = stTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(level, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += dir[2];
    player = from;
    const nxt = boxes.slice();
    nxt[nxt.indexOf(from)] = to;
    nxt.sort((a, b) => a - b);
    boxes = nxt;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower
 *  LURD letters (the first direction first on ties). */
function walkTo(level: Level, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (lower-case LURD) from
 *  `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. Null
 * when a room's search gives up or the level is no set of rooms.
 */
function solveByRooms(level: Level): PushSolution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const other = rooms[oj]!;
      for (const b of done[oj] ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub, MAX_STATES_PER_ROOM);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done[ri] = true;
  }
  return { pushes, moves };
}

/** Fallback: a plain A* over single steps (player square + boxes), like the
 *  house baseline, used only when the push search gives up. Its solutions
 *  take the fewest moves, not necessarily the fewest pushes, but a solved
 *  level still earns partial points. */
function solveByMoves(level: Level): string {
  const { w, wall, goals } = level;
  const goal = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  const boxes0 = level.boxes.slice().sort((a, b) => a - b);
  const dist = (b: number): number => {
    let m = Infinity;
    for (const g of goals) {
      const d = Math.abs((g % w) - (b % w)) + Math.abs(Math.floor(g / w) - Math.floor(b / w));
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => `${p}:${bs.join(',')}`;
  const sp: number[] = [level.player];
  const sb: number[][] = [boxes0];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(level.player, boxes0), 0]]);
  const heap: [number, number, number][] = [[est(boxes0), 0, 0]];
  let seq = 0;
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const push = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const pop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < MAX_STATES_MOVES) {
    const [, g, s] = pop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goal[b])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t]! + moves;
      return moves;
    }
    expanded++;
    const p = sp[s]!;
    for (const [dx, dy, letter] of DIRS) {
      const d = dy * w + dx;
      const q = p + d;
      if (wall[q]) continue;
      let nb = bs;
      const at = bs.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || bs.includes(q + d)) continue;
        nb = bs.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
      }
      const k = key(q, nb);
      if (best.has(k)) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(letter);
      best.set(k, t);
      push([g + 1 + est(nb), g + 1, seq++]);
    }
  }
  return '';
}

function solve(text: string): string {
  const level = parseLevel(text);
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePushOptimal(level, MAX_STATES_WHOLE);
  if (whole !== null) return whole.moves;
  return solveByMoves(level);
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
