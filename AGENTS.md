# Agent Instructions

Follow all repository instructions in `.agents.md`.

In particular, never execute `sbatch`, `srun`, `scancel`, or any other command
that submits, starts, attaches to, restarts, requeues, cancels, or mutates a
Slurm job. Prepare and validate scripts, then give the exact execution commands
to the user to run manually. Read-only scheduler inspection is allowed.
