from typing import Final

import numpy as np

# Constants
CHECKSUM_EXT: Final[str] = ".sha256"
DATASET_OBJS: Final[list] = ["X", "y", "length"]
EPS: Final[float] = np.finfo(float).eps
OBJ_EXT: Final[str] = ".pt"
TQDM_FORMAT: Final[
    str
] = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

# UEA/UCR
UEA_DOWNLOAD_URL: Final[str] = "https://www.timeseriesclassification.com/Downloads/"

# PhysioNet 2012
PHYSIONET_2012_DATASETS: Final[dict] = {
    "set-a": "https://physionet.org/files/challenge-2012/1.0.0/set-a.zip?download",
    "set-b": "https://physionet.org/files/challenge-2012/1.0.0/set-b.zip?download",
    "set-c": "https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download",
}
PHYSIONET_2012_OUTCOMES: Final[list] = [
    "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download",
    "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download",
    "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download",
]
PHYSIONET_2012_VARS: Final[list] = [
    "Hours",
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "Mg",
    "MAP",
    "MechVent",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
    "Age",
    "Gender",
    "Height",
    "ICUType1",
    "ICUType2",
    "ICUType3",
    "ICUType4",
]

# PhysioNet 2019
PHYSIONET_2019_DATASETS: Final[dict] = {
    "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
    "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
}
