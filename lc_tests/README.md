# Tests of the tools for Napari / Fiji

Declarations: `src/guessthecondition_lc_tools/`. The tools run in the app's own environment through the LabConstrictor tools worker.

    python lc_tests/make_fixtures.py lc_tests/fixtures      # a small experiment and an empty results folder
    labconstrictor-tools check --module guessthecondition_lc_tools --pythonpath src
    labconstrictor-tools test  --module guessthecondition_lc_tools --pythonpath src --cases lc_tests/cases.json

`cases.json` plays a whole game in order (the state is in the results folder): prepare it, show the first image, guess, guess with a
wrong name and by number, undo, resume with a higher percentage, finish, analyse, and the errors that are explained.
