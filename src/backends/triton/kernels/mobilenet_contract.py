"""Person A's MobileNet/Triton interface contract for M1-M5.

The constants in this file let Person A run and present kernel work without
waiting for Person B's final training/backpropagation work. Values marked with
``PERSON_B_MUST_REPLACE`` are dummy placeholders and must be replaced with
Person B's frozen MobileNet/CuPy handoff values before final integration.
"""

from __future__ import annotations

from dataclasses import dataclass


PERSON_B_MUST_REPLACE = "PERSON_B_MUST_REPLACE_BEFORE_M6_FINAL_INTEGRATION"

INPUT_SHAPE = (1, 3, 64, 64)
BATCH_SIZES = (1, 8, 16, 32, 64, 128)
DTYPE = "float32"

# These known shapes come from the current MobileNetV2-GTSRB architecture.
FINAL_FEATURE_SHAPE = (1, 1280, 4, 4)
POOL_OUTPUT_SHAPE = (1, 1280, 1, 1)
FLATTEN_SHAPE = (1, 1280)
CLASSIFIER_INPUT_FEATURES = 1280
NUM_CLASSES = 43
CLASSIFIER_WEIGHT_SHAPE = (NUM_CLASSES, CLASSIFIER_INPUT_FEATURES)
CLASSIFIER_BIAS_SHAPE = (NUM_CLASSES,)
LOGITS_SHAPE = (1, NUM_CLASSES)

# DUMMY LINES FROM PERSON A:
# Person B must replace these with actual GTSRB preprocessing statistics.
# They are intentionally ImageNet-style placeholders so nobody mistakes them
# for measured GTSRB values.
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)  # PERSON_B_MUST_REPLACE
NORMALIZATION_STD = (0.229, 0.224, 0.225)  # PERSON_B_MUST_REPLACE

# DUMMY LINES FROM PERSON A:
# Person B must replace these with exact frozen feature indices/layer names and
# saved reference tensors/logits from the CuPy or PyTorch reference path.
TARGET_FEATURE_INDICES = (0, 2, 6, 13, 18)  # PERSON_B_MUST_REPLACE
TARGET_LAYER_NAMES = (
    "features[0]",   # PERSON_B_MUST_REPLACE with exact agreed name if different
    "features[2]",   # PERSON_B_MUST_REPLACE with exact agreed name if different
    "features[6]",   # PERSON_B_MUST_REPLACE with exact agreed name if different
    "features[13]",  # PERSON_B_MUST_REPLACE with exact agreed name if different
    "features[18]",  # PERSON_B_MUST_REPLACE with exact agreed name if different
    "pooling",
    "classifier",
)
REFERENCE_BATCH_PATH = PERSON_B_MUST_REPLACE
REFERENCE_OUTPUTS_PATH = PERSON_B_MUST_REPLACE
WEIGHTS_OR_CHECKPOINT_PATH = PERSON_B_MUST_REPLACE

# Person A implemented both ReLU and ReLU6 forward kernels; Person B/C still
# need to confirm whether M6 replaces MobileNet activations or only compares
# them layer-by-layer.
ACTIVATION_REPLACEMENT = "relu6_or_layer_comparison_pending_person_b_person_c_decision"

# Triton kernels accept torch.Tensor. Person C may centralize the final bridge,
# but Person A provides DLPack helpers for M5 discussion and experiments.
INTEROP_DECISION = "dlpack_recommended_but_pending_person_b_person_c_confirmation"


@dataclass(frozen=True)
class MissingPersonBHandoff:
    """One missing Person B artifact needed after Person A's M4 work."""

    item: str
    why_person_a_needs_it: str
    required_for: str


MISSING_PERSON_B_HANDOFF = (
    MissingPersonBHandoff(
        item="Measured GTSRB normalization mean/std",
        why_person_a_needs_it="Use the same preprocessing as the MobileNet/CuPy reference path.",
        required_for="M3 interface freeze and M5 presentation correctness story",
    ),
    MissingPersonBHandoff(
        item="Exact target layer/block names or feature indices",
        why_person_a_needs_it="Map Triton kernels to the same tensors produced by Person B's model.",
        required_for="M3 interface freeze and M6 integration",
    ),
    MissingPersonBHandoff(
        item="Sample validation batch or loader code",
        why_person_a_needs_it="Run repeatable layer comparisons instead of synthetic-only tests.",
        required_for="M3/M5 demo material",
    ),
    MissingPersonBHandoff(
        item="CuPy and/or PyTorch reference outputs for selected layers",
        why_person_a_needs_it="Check Triton outputs against Person B's actual forward path.",
        required_for="M4 evidence with real tensors and M5 presentation",
    ),
    MissingPersonBHandoff(
        item="Weights/checkpoint or seeded random-weight reference",
        why_person_a_needs_it="Make logits and benchmark comparisons reproducible.",
        required_for="M5 if showing model outputs; M6/M7 for final comparisons",
    ),
    MissingPersonBHandoff(
        item="Final CuPy/Torch/DLPack handoff decision",
        why_person_a_needs_it="Connect torch.Tensor Triton kernels to Person B's CuPy arrays safely.",
        required_for="M6 hybrid or layer-comparison path",
    ),
)
