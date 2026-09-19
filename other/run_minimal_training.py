"""Start a small, end-to-end CPU training run for the Amazons project.

Use ``python -m train.train`` when choosing custom settings or resuming a
checkpoint.  This entry point deliberately uses the 3x3 teaching board so a
new environment can verify the whole reinforcement-learning loop quickly.
"""

from __future__ import annotations

import sys

from train.train import main


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--board-size", "3",
            "--iterations", "2",
            "--games-per-iteration", "2",
            "--simulations", "4",
            "--train-batches", "4",
            "--batch-size", "16",
            "--replay-capacity", "1_000",
            "--evaluation-games", "2",
            "--channels", "16",
            "--residual-blocks", "1",
            "--device", "cpu",
            "--no-amp",
            "--checkpoint", "checkpoints/minimal_3x3.pt",
        ]
    )
    main()
