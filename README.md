# SQL-RL

SQL-RL is a lightweight SQL debugging / repair RLVR project.

The first implementation step is a minimal verifiable loop:

1. Load BIRD-Critic / SIX-GYM style SQL debugging records.
2. Build SQL repair prompts.
3. Evaluate predicted SQL with SQLite execution and Python test cases.
4. Convert the evaluation result into a reward signal.

The BIRD-RL critic evaluator is vendored under `sql_rl/vendor/bird_critic_eval/`
so behavior that should match BIRD-RL is copied instead of rewritten.

