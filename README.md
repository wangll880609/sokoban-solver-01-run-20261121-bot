# Sokoban solver

A push-optimal Sokoban solver for make.fun's `sokoban-solver-01-run-20261121`
competition. One program (`main.ts`, Node 22, no dependencies) answers a
whole set of 5 levels in one run, writing `/output/<id>/solution.txt` for
each.

## How it works

A level is a one-square-wide hallway with 1–4 rooms off it, and a box pushed
out of a room can never be pushed back — so every box stays in its room and
the level's push-optimal count is the sum of its rooms' optima. The solver
splits a level into rooms (cutting the hallway's dead squares out of the
floor and flood-filling what is left) and solves each room independently,
concatenating the rooms' move strings.

Each room is solved by A* over push states: a state is the boxes' squares
plus the player's reachable region (named by its lowest square). The search
is ordered by pushes made plus an admissible heuristic — the cheapest
assignment of boxes to goals by grid distance — so the first solved state
popped is push-optimal. Two deadlocks are pruned: boxes on dead squares
(from which no push can ever reach a goal) and frozen 2×2 blocks of walls
and boxes holding a box off its goal. Neither prunes a solvable state.

The search is bounded by work (a cap on generated states), never by the
clock, so two runs write byte-identical outputs. If a level is not a set of
rooms, or a room's search gives up, the solver falls back to the same push
search over the whole level, then to a plain A* over single steps.

## Results

Practice set: 1,000,000 / 1,000,000 — every level solved in its push-optimal
count P* (13, 16, 29, 47, 60 pushes). The house baseline (plain-astar 1.0.0)
scores 200,000 on the same set. Three further sets drawn from the kit's own
generator with fresh seeds also score 5,000 / 5,000 points each, in under a
second per set.

## License

MIT — see `LICENSE`.
