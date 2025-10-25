"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
import dataclasses
from dataclasses import dataclass, field
from argparse import Namespace

@dataclass 
class Configuration:
    """Configuration class for the project."""

    DATA_FOLDER: str = os.path.join("..", "data")
    PROCESSED_FOLDER: str = os.path.join("..", "data", "processed")
    LOGS_FOLDER: str = os.path.join("..", "logs")
    MODELS_FOLDER: str = os.path.join("..", "models")
    EXPERIMENT_NAME: str = "IACALFRIT"  # DO NOT TOUCH

    create_folders: bool = False  # Whether to create folders at init

    # =================================== DATA ======================================
    train_data: str = os.path.join(DATA_FOLDER, "dataset_train.csv")
    test_data: str = os.path.join(DATA_FOLDER, "dataset_test.csv")


    label: str = "genre"
    columns: list = field(default_factory=list)

    val_split: float = 0.15

    # ================================== EXPERIMENTS ======================================
    gemini_prediction_path: str = os.path.join(LOGS_FOLDER, "gemini_prediction.csv")
    gemini_results_path: str = os.path.join(LOGS_FOLDER, "gemini_results.csv")
    gemini_scores_path: str = os.path.join(LOGS_FOLDER, "gemini_scores.json")


    # ================================== VARIABLES ======================================
    seed:     int = 42
    batch_size: int = 64

    learning_rate: float = 2.5e-4
    total_timesteps: int = 25_000

    torch_deterministic: bool = True
    cuda:                bool = True


    def __post_init__(self):
        ...
