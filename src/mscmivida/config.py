import platform
from os import environ, getenv
from pathlib import Path

import dotenv

# Project root
project_dir = Path(__file__).resolve().parents[2]

# Load the environment variables from the `.env` file, overriding any system environment variables
dotenv.load_dotenv(project_dir / ".env", override=True)

# Load secrets from the `.secrets` file, overriding any system environment variables
dotenv.load_dotenv(project_dir / ".secrets", override=True)

# Some common paths
_reports_dir = str(getenv("DIR_REPORTS"))
report_dir = project_dir / _reports_dir / "figures"

_figures_dir = str(getenv("DIR_FIGURES"))
figures_dir = project_dir / _figures_dir

_models_dir = str(getenv("DIR_MODELS"))
models_dir = project_dir / _models_dir

_notebook_dir = str(getenv("DIR_NOTEBOOKS"))
notebook_dir = project_dir / _notebook_dir

_data_dir = Path(str(getenv("DIR_DATA")))
data_dir = project_dir / _data_dir

_data_raw_dir = Path(str(getenv("DIR_DATA_RAW")))
data_raw_dir = project_dir / _data_raw_dir

_data_interim_dir = Path(str(getenv("DIR_DATA_INTERIM")))
data_interim_dir = project_dir / _data_interim_dir

_data_processed_dir = Path(str(getenv("DIR_DATA_PROCESSED")))
data_processed_dir = project_dir / _data_processed_dir

# CUDA Enable
_ENABLE_CUDA = True
if not _ENABLE_CUDA:
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    environ["USE_CPU"] = "1"

# Hydra
environ["HYDRA_FULL_ERROR"] = "1"

# log to mlflow
LOG_TO_MLFLOW = False

_IS_WINDOWS = platform.system() == "Windows"

# Utility constants related to the MIT-BIH loader
MIT_BIH_PATH = "/net/sharedfolders/datasets/MOTION/mit-bih-arrhythmia-database-1.0.0/"
AAMI = ["N", "SVEB", "VEB", "F", "Q"]
AAMI_N = ["N", "L", "R"]
AAMI_SVEB = ["A", "a", "J", "S", "e", "j"]
AAMI_VEB = ["V", "E"]
AAMI_F = ["F"]
AAMI_Q = ["P", "/", "f", "u"]
AAMI_ALL = AAMI_N + AAMI_SVEB + AAMI_VEB + AAMI_F + AAMI_Q

# EOF
