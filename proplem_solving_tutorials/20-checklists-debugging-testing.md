## 20 — Checklists, Debugging & Testing

Design checklist
- Clarify constraints and edge cases up front
- Choose correct DS: hash vs tree vs heap vs array
- Time and space estimates before coding
- Validate invariants (monotonicity, balance conditions, feasibility)

Implementation checklist
- Boundary indices (l, r, mid) and off-by-one
- Null/empty inputs; single-element arrays; all equal values
- Overflow: use `long` if sums/products may exceed `int`
- Avoid O(N^2) nested loops on large N

Testing checklist
- Tiny cases: size 0/1/2; equal elements; extremes
- Randomized tests (when possible) to compare slow vs fast
- Deterministic fixtures for known tricky cases

Debugging tactics
- Print small trace for minimized example
- Binary search the bug: comment halves of logic or use asserts
- Check assumptions: are invariants really maintained?

Performance profiling
- Measure via `System.nanoTime()` around hot loops
- Replace boxing with primitives; prefer arrays where possible
- Pre-allocate buffers to avoid reallocation in critical paths

Reusable helpers
- `assert` and parameter validation
- Fast I/O for large inputs
- Utility builders for graphs, trees, tries, and DSU


