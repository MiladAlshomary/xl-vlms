from typing import Any, Callable, Dict

import torch
import numpy as np
import argparse
from tqdm import tqdm

from analysis.feature_decomposition import *


def compute_causal_effect(
    model_class: Any,
    decomposition_results: Dict[str, Any],
    dataloader: Any,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:
    """
    The function compute the change the liklihood of predicting predicted_token if each of 
    the concepts (vectors) was removed from latent space
    
    :param model_class: The model wrapper class containing the model and tokenizer
    :param decomposition_results: Dictionary containing concepts, activations, and metadata
    :param dataloader: Dataloader to iterate over the dataset
    :param predicted_token: The target token string to measure likelihood for
    :param device: Torch device
    :param logger: Optional logger
    """

    predicted_token = args.token_of_interest

    concepts = decomposition_results["concepts"]
    activations = decomposition_results["activations"]
    module_to_decompose = args.module_to_decompose

    # Map image paths to their index in the activations matrix
    image_paths_ref = decomposition_results.get("image_to_info", [])
    print(len(image_paths_ref))
    path_to_idx = {path: i for i, path in enumerate(image_paths_ref.keys())}

    model = model_class.get_model()
    tokenizer = model_class.get_tokenizer()
    model.eval()

    # Get the ID for the predicted token
    target_token_ids = tokenizer.encode(predicted_token, add_special_tokens=False)
    if not target_token_ids:
        raise ValueError(f"Token '{predicted_token}' not found in vocabulary")
    target_token_id = target_token_ids[0]

    results = {"causal_effects": [], "base_probs": [], "image_paths": []}

    for batch in tqdm(dataloader, desc="Computing Causal Effect"):
        # Assuming batch size 1 for precise mapping
        image_path = batch["image"][0]
        text = batch["text"][0]

        if image_path not in path_to_idx:
            continue

        idx = path_to_idx[image_path]
        sample_activations = activations[idx]  # (num_concepts,)

        inputs = model_class.preprocessor(
            instruction=text,
            image_file=image_path,
            response="",
            generation_mode=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 1. Base prediction (no intervention)
        with torch.no_grad():
            outputs = model(**inputs)
            # Probability of target token at the last position
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            base_prob = probs[target_token_id].item()

        effects = []

        # 2. Intervention: Remove each concept one by one
        for k in range(concepts.shape[0]):
            concept_vec = torch.tensor(concepts[k]).to(device)
            coeff = sample_activations[k]
            
            # To remove the concept, we subtract its contribution: -1 * coeff * vector
            intervention_vec = -1 * coeff * concept_vec

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                
                # Apply intervention to the latent representation
                h = h + intervention_vec.to(h.device).to(h.dtype)
                
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            # Register hook
            hook_handle = None
            for name, module in model.named_modules():
                if name == module_to_decompose:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
            
            if hook_handle is None:
                if logger: logger.warning(f"Module {module_to_decompose} not found.")
                break

            # Run model with intervention
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
                new_prob = probs[target_token_id].item()

            # Causal Effect = Prob(removed) - Prob(base)
            effects.append(new_prob - base_prob)
            hook_handle.remove()

        results["causal_effects"].append(effects)
        results["base_probs"].append(base_prob)
        results["image_paths"].append(image_path)
        results['concepts_to_style_contribution'] = np.mean(results['causal_effects'], axis=0)

    return results