<div align="center">

<img src="app/logo/logo.png" width="360" alt="Guess the Condition logo"/>

# Guess the Condition

**Can you tell experimental conditions apart just by looking at microscopy images?**

Guess the Condition shows you images without revealing which condition each belongs to. You guess the condition, then see how often you were right, which conditions you confused, and how your results varied between biological repeats.

[![Open the interactive notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CellMigrationLab/GuessTheCondition/blob/main/notebooks/GuessTheCondition/GuessTheCondition.ipynb)

[Download the desktop app](https://github.com/CellMigrationLab/GuessTheCondition/releases) · [View the example notebook](notebooks/GuessTheCondition_API_Example/GuessTheCondition_API_Example.ipynb)

</div>

## Why use it?

Sometimes a difference looks obvious in one field of view but is much harder to spot across a whole experiment. This is a way to check whether someone can recognise the conditions **without knowing which image came from which condition**.

You see the available condition names, but not the true label or filename of the image on screen. The app records your answer and how long you took. It then shows the answers for each biological repeat, rather than treating all fields of view as independent experiments.

It is a check of **visual distinguishability**, not a substitute for measuring the phenotype or independently validating a biological effect.

## Get started

**In Google Colab:** [open the interactive notebook](https://colab.research.google.com/github/CellMigrationLab/GuessTheCondition/blob/main/notebooks/GuessTheCondition/GuessTheCondition.ipynb). Run the first cell to install the package and connect Google Drive. Put your images in Drive first. Colab runs on Google's servers, so consider your data-sharing requirements.

**On your computer:** [download an installer](https://github.com/CellMigrationLab/GuessTheCondition/releases) and follow the [installation guide](.tools/docs/download_executable.md). The published release has installers for Windows (x86-64), macOS (Apple Silicon and Intel), and Linux (x86-64); these files have not all been independently tested here. Launch the app, open the **GuessTheCondition** notebook from its JupyterLab welcome page, and run the cells in order. You do not need to write Python code.

**No images to try yet?** The [Python API example](notebooks/GuessTheCondition_API_Example/GuessTheCondition_API_Example.ipynb) generates a small synthetic experiment and runs a simulated player. It illustrates the results but is not the interactive guessing game.

## Prepare your images

Use TIFF images (`.tif` or `.tiff`) in the following folder structure:

```text
Experiment/
├── Control/
│   ├── R1/
│   │   ├── FOV1.tif
│   │   └── FOV2.tif
│   ├── R2/
│   └── R3/
└── Treated/
    ├── R1/
    ├── R2/
    └── R3/
```

The first folder level names the **conditions**; the second names the **biological repeats**. You need at least two conditions. Use a separate repeat identifier for each independent experiment, not for every image or field of view. The app groups folders with the same repeat name together across conditions. To compare accuracy across repeats, each repeat needs images from at least two conditions.

Three or more independent, usable repeats are recommended for the main statistical comparison. A small number of images per repeat gives an imprecise result, even if the overall image count is large.

The game accepts 2D, multichannel and stack/time-series TIFF images. Channels are displayed separately, and stacks or time series are normally shown as maximum-intensity projections. **Check that this display suits your question:** if TIFF axis metadata is ambiguous, a short stack can be interpreted as channels instead.

## Play and see the results

1. **Step 0 — Initialise:** Run the setup cell in the notebook.
2. **Step 1 — Prepare:** Enter the experiment folder, a results folder and an identifier for yourself. Choose what percentage of the images to guess (20% by default), then click **Prepare the game**.
3. **Step 2 — Guess:** Click **Start / continue**, choose a condition and click **Submit guess**. Images are shown in a shuffled order spread across conditions and repeats. Use **Undo last guess** to correct a mistake.
4. **Step 3 — Analyse:** Run the analysis cell whenever you have guesses to examine. It displays the summary, figures and tables and writes a PDF report.

Your files are saved under `<results folder>/<your identifier>/`:

- `game_results.csv` — one row per image, including the true condition, your guesses and decision times.
- `session.json` — information needed to continue your game.
- `analysis/analysis_results.pdf` — the report; the same folder also contains figures and tables.

You can close the notebook and continue later using the same results folder and identifier, provided the original experiment is still available at its saved path. **Colab and desktop do not automatically synchronise files or paths.** Choosing **Start again** creates backups of the previous results.

## How to read the answer

The report compares your accuracy with a simple baseline: always guessing the most common condition in the experiment. It includes accuracy per biological repeat and per condition, a confusion matrix showing mistaken identities, and decision-time plots.

With **at least three usable biological repeats**, the main statistical comparison is based on each repeat's accuracy relative to its own chance level. With **one or two**, the app reports an exploratory test of the images shown, not evidence that the result will reproduce in other experiments. Tests pooling individual images are also exploratory: multiple fields of view do not become independent biological repeats.

An above-chance result means that a person could recognise a pattern in these images under these viewing conditions. It does not establish what caused that pattern. Conversely, a result close to chance does not prove the conditions are identical.

**Important display limitation:** Each channel is contrast-stretched independently (0.01–99.99th percentiles). Absolute brightness differences are therefore not preserved. Do not use the game to test whether conditions differ only in fluorescence intensity. Review the displayed images before interpreting the result.

Unreadable images are skipped and flagged during play; check any warnings before trusting the report.

## Other ways to use it: Napari, Fiji and QuPath

Besides the notebook, the same game runs inside **Napari**, **Fiji** and **QuPath** through the LabConstrictor tools bridge: the desktop installer registers the app's tools (`src/guessthecondition_lc_tools`), and each host lists **Guess the Condition** with one form per tool. The images are shown by the host (Napari layer, Fiji window, QuPath preview) and the game runs in the app's own environment, so the hosts never import it. They use the same experiment and results folders as the notebook, so a game started in one place can be continued in another.

| Tool | What it does |
|---|---|
| **Create a demo experiment** | Makes a small synthetic experiment (two conditions, three repeats) in a folder you choose, to try the game |
| **Prepare the game** | Scans your experiment folder and starts or resumes your game; you choose the share of images to guess |
| **Show the conditions** | Lists the condition names you can use as a guess |
| **Play a round** | Records your guess (a condition name or its number) and shows the next image; the true condition is never shown |
| **Undo my last guess** | Forgets the last guess, and that image is shown again |
| **Analyse the results** | Accuracy against chance across the biological repeats, per condition, the confusion matrix and decision times, plus a PDF report |

How to install and use the Napari, Fiji and QuPath plugins is described in the [LabConstrictor-Tools documentation](https://github.com/CellMigrationLab/LabConstrictor-Tools/tree/main/docs) and in the READMEs of [LabConstrictor-napari](https://github.com/CellMigrationLab/LabConstrictor-napari), [LabConstrictor-Fiji](https://github.com/CellMigrationLab/LabConstrictor-Fiji) and [LabConstrictor-QuPath](https://github.com/CellMigrationLab/LabConstrictor-QuPath).

If you want to automate the workflow, start with the [Python API example notebook](notebooks/GuessTheCondition_API_Example/GuessTheCondition_API_Example.ipynb).

For installation or usage problems, [open an issue](https://github.com/CellMigrationLab/GuessTheCondition/issues).

## Licence and credits

MIT, see [LICENSE](LICENSE). Developed by the Cell Migration Lab. Packaged with [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor).
