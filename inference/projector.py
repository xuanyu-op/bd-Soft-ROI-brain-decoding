import torch
import warnings


def load_mm_projector_weights(projector_model, weights_path):
    # Load the weight file for mm_projector
    print(f"Loading MM Projector from {weights_path}...")
    weights = torch.load(weights_path, map_location='cpu')

    if 'model_state_dict' in weights:
        weights = weights['model_state_dict']

    cleaned_weights = {}

    # Keep only parameters related to mm_projector and remove the corresponding prefixes
    for k, v in weights.items():
        if k.startswith('model.mm_projector.'):
            cleaned_weights[k.replace('model.mm_projector.', '')] = v
        elif k.startswith('mm_projector.'):
            cleaned_weights[k.replace('mm_projector.', '')] = v

    # Read the parameter names of the current model for later consistency checking
    model_keys = projector_model.state_dict().keys()

    # Find parameters required by the current model but missing from the loaded weights
    missing_keys = [k for k in model_keys if k not in cleaned_weights]

    # Find parameters in the loaded weights but not used by the current model
    unexpected_keys = [k for k in cleaned_weights if k not in model_keys]

    # Raise a warning if the parameters do not match exactly
    if missing_keys or unexpected_keys:
        warnings.warn(
            f"MM-Projector weight mismatch detected.\n"
            f"Missing: {missing_keys}\n"
            f"Unexpected: {unexpected_keys}"
        )

    # Load the weights in non-strict mode so partial mismatches are allowed
    projector_model.load_state_dict(cleaned_weights, strict=False)