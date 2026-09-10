# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Running anything that touches isaacgym

Use the `hvgym` conda env (Python 3.8). Two variables are required and are not set for you:

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # isaacgym bindings
    export PATH=$CONDA_PREFIX/bin:$PATH                          # ninja, for the C++ extension load

Without the first you get `libpython3.8.so.1.0: cannot open shared object file`; without the
second, `RuntimeError('Ninja is required to load C++ extensions')` surfacing as a hydra
`Error getting class ... IsaacGym`.

**If you work in a git worktree, set `PYTHONPATH` to your worktree.** ASAP is pip-installed
editable into `hvgym` pointing at the primary checkout, so an unqualified run executes that
tree's code, not yours. Print `humanoidverse.__file__` and confirm it before trusting a run.

## The motion-tracking reference lead

`LeggedRobotMotionTracking._reference_lead_steps()` is the single place that defines how far
the tracking reference leads the simulated state. It returns 1 for motion tracking (the
callback runs after `episode_length_buf += 1`, so the reference is the next target to steer
toward). System-identification environments must override it to 0, as `DeltaA_OpenLoop` does:
there the reset anchors the robot at the current step, so scoring one step further on asks
one dt of physics to cover two dt of recording. Read the docstrings on both before changing
either; the convention is load-bearing for already-deployed stage (a) policies.

## replay_real_rollouts.py scores a random subset unless you pass --seed

The motion library is loaded during env construction with `random_sample=True`, drawing from
an unseeded RNG, and `set_is_evaluating()` only runs afterwards. Two runs launched back to
back therefore score **different clips**, and randomised start times diverge on top of that.
Pass `--seed N` to any comparison, or the numbers are not comparable. Verify rather than
assume: the saved `ref_dof` depends only on the clips and start times, so bit-identical
`ref_dof` across conditions proves identical data.

## Reproducing a past training run

Derive the command from that run's own `.hydra/.hydra/overrides.yaml` and `config.yaml`, not
from the README. Several August 2026 run configs point `robot.motion.motion_file` at a
deleted `/tmp/claude-1000/...` path; the permanent copy is under
`humanoidverse/data/motions/g1_29dof_anneal_23dof/real_dab/`. To prove a reproduction before
spending GPU hours, compose the config with `--cfg job` and diff it against the old run's
saved `config.yaml`.

Motion data, logs and checkpoints are gitignored, so a clean checkout does not contain them.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
