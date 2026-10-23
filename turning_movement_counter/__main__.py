"""Entry point: `python -m turning_movement_counter` opens the window,
`python -m turning_movement_counter count ...` counts a video from the command line.

Absolute imports on purpose: PyInstaller runs this file as a top-level script,
where a relative import has no parent package and fails.
"""

import os
import sys

# The windowed exe starts with no stdout/stderr; Ultralytics and tqdm write to them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        from turning_movement_counter.cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    from turning_movement_counter.ui.main_window import main

    main()
