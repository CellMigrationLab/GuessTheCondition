<div align="center">

<img src="app/logo/logo.png" width="420" alt="Guess the Condition logo"/>

# Guess the Condition

**Can the experimental conditions be told apart by looking at the images?** Guess the Condition shows you images without their condition, asks you to name it, and tells you whether you did better than chance *across the biological repeats*.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CellMigrationLab/GuessTheCondition/blob/main/notebooks/GuessTheCondition/GuessTheCondition.ipynb)

</div>

---

## What it is for

Before investing in the quantification of a phenotype, ask a human: is it visible, and is it reproducible? You are presented with randomised images (spread over the conditions and the repeats), you guess each one's condition, and the tool records your guesses and your decision times, then reports:

- whether the conditions can be told apart **across biological repeats** (the repeat is the replicate), with n and the 95% interval;
- the accuracy of every repeat and of every condition, against chance;
- the confusion matrix and the pairs of conditions that get mistaken for each other;
- the decision times (a SuperPlot);
- a PDF report with the text, the figures and the tables.

This repository brings together the former [Napari plugin](https://github.com/CellMigrationLab/napari-guess-the-condition) and [Colab notebook](https://github.com/CellMigrationLab/jupyter-guess-the-condition) in one [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor) app: one package, one set of results, four ways to play.

## Prepare your images

One folder per condition, inside it one folder per biological repeat, inside it the TIFF images (`.tif` or `.tiff`):

```
Experiments
├── Control
│   ├── R1   (FOV1.tif, FOV2.tif, ...)
│   └── R2
└── Mutant
    ├── R1
    └── R2
```

The folder names are the *condition* and the *biological repeat* of the images inside. 2D images, multi-channel images (shown side by side) and z-stacks or time series (shown as a maximum projection) all work. Every channel is contrast-stretched on its own (0.01-99.99 percentiles, like ImageJ's "Auto"), so the conditions have to be told apart by what the images show, not by their overall brightness.

## Four ways to play (one game, any of them)

Your game lives in a folder with your name inside the results folder (`game_results.csv` and `session.json`), so you can **stop in one and resume in another**.

### 1. The desktop app (Windows, macOS, Linux)

Download the installer from the [Releases page](https://github.com/CellMigrationLab/GuessTheCondition/releases) (see the [installation guide](.tools/docs/download_executable.md)), install it, and open **Guess the Condition** from your desktop or Start menu. The notebook walks you through three steps: prepare the game, guess, analyse. Apple-silicon Macs are supported; Intel Macs are not.

### 2. Google Colab

Press the *Open in Colab* badge above: the notebook installs the package and mounts your Google Drive.

### 3. Napari and Fiji (tools)

The installer also registers five tools for the [LabConstrictor tools bridge](https://github.com/CellMigrationLab/LabConstrictor-Tools), so that [Napari](https://github.com/CellMigrationLab/napari-labconstrictor) (Plugins > LabConstrictor tools) and [Fiji](https://github.com/CellMigrationLab/LabConstrictor-Fiji) (Plugins > LabConstrictor > LabConstrictor Tools...) show them as forms, and so that the command line can run them:

| Tool | What it does |
|---|---|
| **Create a demo experiment** | Makes a small synthetic experiment (two conditions, three repeats) to try the game before using your own images. |
| **Prepare the game** | Scans the experiment folder and starts (or resumes) your game. |
| **Play a round** | Records your guess for the image on screen (a condition name or its number; leave it unset the first time) and shows the next image. The condition is never shown. |
| **Undo my last guess** | Forgets the last guess; that image is shown again. |
| **Analyse the results** | The answer, tables and the PDF report. |

In Napari, the guess field keeps its value between runs: change it for every image. The tool tells you what it recorded every time, and **Undo my last guess** takes a mistake back.

From a terminal (use the Python of the installed app):

```
<install folder>/bin/python -m labconstrictor_tools run GuessTheCondition create_demo_experiment output_folder=.      # optional, to try the game
<install folder>/bin/python -m labconstrictor_tools run GuessTheCondition prepare_game experiment_folder=Experiments results_folder=Results user_name=Ana
<install folder>/bin/python -m labconstrictor_tools run GuessTheCondition play_round results_folder=Results user_name=Ana
<install folder>/bin/python -m labconstrictor_tools run GuessTheCondition play_round results_folder=Results user_name=Ana guess=Mutant
```

### 4. Python

```python
from guessthecondition import Session, analyze

session = Session.create("Experiments", "Results", user_name="Ana", percentage_to_test=20)
item = session.serve()        # {"number", "of", "path"}: nothing that names the condition
session.answer("Mutant")      # a condition name or its number; records the decision time
analysis = analyze(session.answered())
print(analysis.text())
```

`notebooks/GuessTheCondition_API_Example` runs a whole game with a simulated player. `pip install "guessthecondition @ git+https://github.com/CellMigrationLab/GuessTheCondition"` installs the package alone.

## How the results are read

The rules are those of the other MorphoBricks tools ([MorphoCoverage](https://github.com/CellMigrationLab/MorphoCoverage)):

- **The biological repeat is the replicate.** Images of one repeat are not independent. The answer is the accuracy of each repeat compared with chance across the repeats (a one-sided t-test on accuracy minus chance, from 3 repeats). With one or two repeats the numbers are shown but no verdict is given.
- **Chance** is the accuracy of someone who ignores the images and always names the most common condition (1 / number of conditions when balanced, as the order of the images makes them).
- The test on **all images pooled** (exact binomial) and the test of whether the repeats differ are shown as *exploratory*: they count every image as independent, so their false-positive chance is higher than stated.
- ✅ the conditions can be told apart across repeats · ⚠️ a trend only · ➖ no conclusion possible, or not above chance · ❌ check the data first. It never says "confirmed", and a null result is said as a result.
- Nothing is pooled silently: the per-repeat and per-condition tables come with the pooled figures.

## Development

```
pip install -e . pytest labconstrictor-tools
pytest tests                                             # the package, the notebook helpers and the tool declarations
python lc_tests/make_fixtures.py lc_tests/fixtures
labconstrictor-tools test --module guessthecondition_lc_tools --pythonpath src --cases lc_tests/cases.json
```

`src/guessthecondition` is the API (`dataset`, `images`, `session`, `analysis`, `report`); `src/guessthecondition_lc_tools` declares the tools (its name must be `<package>_lc_tools` for the installer to register it) and imports the package only inside the functions.

## Licence and credits

MIT. Created by [Guillaume Jacquemet](https://cellmig.org/) and the Cell Migration Lab. Built with [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor).
