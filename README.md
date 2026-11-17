# Smart Vision Assistant for Visually Impaired People

Group project for the course *Artificial Intelligence with Python* (ESIGELEC).

A camera worn on the chest films a classroom. A fine-tuned **YOLO11** model detects
obstacles (people, chairs, tables, bags, cables...), **MiDaS** estimates their distance,
and the system describes the scene out loud:

> "Warning: cable on the floor, in front of you. A chair on your left, about 1 meter away."

## Team

| Member | Main responsibility |
|---|---|
| Member 1 | Dataset: capture, labeling (CVAT), split |
| Member 2 | Model: training runs, evaluation |
| Member 3 | Application: camera, distance, voice, live demo |

## Classes

`0 person` · `1 chair` · `2 table` · `3 laptop` · `4 pc` · `5 backpack` · `6 handbag` · `7 cable`

## Repository structure

```text
smart-vision-assistant/
├── README.md
├── requirements.txt
├── setup-windows.bat
├── setup-linux.sh
├── scripts/
│   └── 00_check_environment.py   # step 1: environment check
├── notebooks/                    # experiments and evaluation
├── data/
│   ├── raw/                      # original photos (not on GitHub)
│   └── dataset/                  # YOLO format: images/ and labels/ (train, val, test)
├── models/                       # trained weights (not on GitHub)
├── runs/                         # YOLO training outputs (not on GitHub)
└── docs/                         # project proposal, presentation
```

Images, labels and model weights are **not** stored on GitHub (see `.gitignore`).
They are shared in the group Drive.

## Setup

Requires **Python 3.11 or newer**.

```bash
git clone <URL of this repository>
cd smart-vision-assistant
```

**Windows**

```bat
setup-windows.bat
```

**Linux**

```bash
bash setup-linux.sh
sudo apt install espeak-ng   # needed for text-to-speech on Linux
```

The setup script creates `.venv`, installs `requirements.txt` and runs the environment check.
The first run downloads YOLO11n (5 MB) and MiDaS (about 100 MB), so internet access is needed.

### Check the environment again

```bat
.venv\Scripts\python.exe scripts\00_check_environment.py            (Windows)
.venv/bin/python scripts/00_check_environment.py                     (Linux)
```

Options: `--no-camera` (no webcam), `--no-midas` (no internet),
`--source 1` or `--source http://<phone-ip>:4747/video` (other camera / phone stream).

### Start JupyterLab

```bat
.venv\Scripts\python.exe -m jupyter lab        (Windows)
.venv/bin/python -m jupyter lab                (Linux)
```

## Credits

- [Ultralytics YOLO11](https://docs.ultralytics.com/) (AGPL-3.0)
- [MiDaS](https://github.com/isl-org/MiDaS) (MIT)
- Course: *Artificial Intelligence with Python*, Prof. Dr.-Ing. Wei Yap Tan
