from pathlib import Path

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "0.0.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
WANDB_PROJECT = "taonet-subnet"
# The uid for this subnet.
SUBNET_UID = 28
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo (1 Gigabyte).
MAX_HUGGING_FACE_BYTES = 1 * 1024 * 1024 * 1024
# The maximum parameter size allowed for models.
MAX_MODEL_PARAMETER_SIZE = 186_000_000
# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.04
# validator score boosting for earlier models.
timestamp_epsilon = 0.005
# validators number of pages to eval over miners on each step.
n_eval_pages = 3
# validator eval batch size.
batch_size = 1
# validator eval sequence length.
sequence_length = 1024
