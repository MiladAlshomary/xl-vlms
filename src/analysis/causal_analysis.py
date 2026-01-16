from typing import Any, Callable, Dict

import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from analysis.feature_decomposition import *
from helpers.utils import setup_hooks, clear_hooks_variables, clear_forward_hooks


def compute_activateions(outputs, model_class: Any, analysis_model: Any, module_to_decompose: str, logger: Callable = None, args: argparse.Namespace = None):

    hook_return_functions, _ = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )

    # Compute activations using hooks
    item = {"model_output": outputs.logits}
    captured_data = {}
    for func in hook_return_functions:
        if func is not None:
            res = func(**item)
            if res:
                captured_data.update(res)

    hidden_state = captured_data.get(module_to_decompose, list(captured_data.values())[0] if captured_data else None)
    hidden_state = hidden_state[args.module_to_decompose]
    rep = hidden_state[-1].float().cpu().numpy()

    sample_activations = project_representations(rep, analysis_model, args.decomposition_method)[0]
    
    clear_hooks_variables()

    return sample_activations
    

def compute_causal_effect(
    model_class: Any,
    decomposition_results: Dict[str, Any],
    dl: DataLoader,
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
    module_to_decompose = args.module_to_decompose
    analysis_model = decomposition_results.get("analysis_model")

    args.modules_to_hook = [[module_to_decompose]]
    args.hook_names = ["save_hidden_states"]


    model = model_class.get_model()
    tokenizer = model_class.get_tokenizer()
    model = model.to(device)
    model.eval()

    # Get the ID for the predicted token
    target_token_ids = tokenizer.encode(predicted_token, add_special_tokens=False)
    if not target_token_ids:
        raise ValueError(f"Token '{predicted_token}' not found in vocabulary")
    target_token_id = target_token_ids[0]

    results = {"causal_effects": [], "base_probs": [], "image_paths": []}

    for batch in tqdm(dl, desc="Computing Causal Effect"):
        # Assuming batch size 1 for precise mapping
        image_path = batch["image"][0]
        text = batch["text"][0]

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


        sample_activations = compute_activateions(outputs, model_class, analysis_model, module_to_decompose, logger, args)

        effects = []
        # 2. Intervention: Remove each concept one by one
        for k in range(concepts.shape[0]):
            concept_vec = concepts[k].clone().to(device)
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

    clear_forward_hooks(model_class.model_)
    return results