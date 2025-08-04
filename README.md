# ⚽ Player Tracing, Ball Control Percentage & Speed Estimation

A computer vision-based football analytics toolkit that tracks players, estimates their speed, and calculates ball control percentage using video data. This project is ideal for performance analysis, tactical breakdowns, and sports science applications.

## 📁 Repository Structure

| Folder | Description |
|--------|-------------|
| `camera_movment_estimator/` | Handles camera motion compensation to improve tracking accuracy |
| `data_augmentation/` | Tools for augmenting training data for robustness |
| `development_and_analysis/` | Scripts for model development and performance evaluation |
| `player_ball_assigner/` | Assigns ball possession to players based on proximity and movement |
| `speed_and_distance_estimator/` | Calculates player speed and movement distance |
| `team_assigner/` | Identifies team affiliation of players |
| `trackers/` | Implements tracking algorithms (e.g., SORT, DeepSORT) |
| `utils/` | Utility functions for preprocessing and data handling |
| `view_transformer/` | Transforms camera view to bird’s-eye perspective |
| `main.py` | Entry point for running the full pipeline |

## 🚀 Features

- 🎯 **Player Tracking**: Detect and track players across frames using advanced tracking algorithms.
- 🏃 **Speed Estimation**: Calculate real-time player speed based on positional changes.
- 🕹️ **Ball Control Percentage**: Estimate how long each player controls the ball during a match.
- 🔄 **Camera Motion Compensation**: Stabilize footage to improve tracking accuracy.
- 🧠 **Team Assignment**: Automatically classify players into teams using visual cues.

## 🧪 Technologies Used

- Python
- OpenCV
- Jupyter Notebook
- Deep Learning (YOLO or similar for detection)
- Tracking algorithms (SORT, DeepSORT)

## 📦 Installation

```bash
git clone https://github.com/ahmed-rabi/player-tracing-and-ball-control-percentage-and-speed-estimation-
cd player-tracing-and-ball-control-percentage-and-speed-estimation-
pip install -r requirements.txt
