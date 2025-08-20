from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from .image_text_model import ImageTextModel

from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.conversation import conv_templates, SeparatorStyle
from share4v.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from share4v.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

__all__ = ["sharegpt4v"]


class ShareGPT4v(ImageTextModel):

    def set_model(
        self,
    ) -> None:

        tokenizer, model, image_processor, _  = load_pretrained_model(
            model_path=self.model_name_or_path,
            model_base=None,
            model_name=get_model_name_from_path(self.model_name_or_path,)
        )

        self.model_ = model
        self.processor_ = image_processor
        self.tokenizer_ = tokenizer

    def get_model(
        self,
    ) -> Callable:

        return self

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.model.transformer

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.model.transformer.ff_out

    def set_processor(
        self,
    ) -> None:

        return

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: str = None,
        response: str = "",
        conv_mode: str ='share4v_v1',
        **kwargs: Any,
    ) -> Dict[str, Any]:

        # 2. Prepare the conversation prompt using the specified conversation mode
        # The prompt needs to be formatted with the special image token placeholder
        query = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"
        print(conv_templates)
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # 1. Load and process the image
        image = Image.open(image_file).convert('RGB')
        # The image processor for LLaVA/Share4V has a specific preprocess method
        image_tensor = self.processor_.preprocess(image, return_tensors='pt')['pixel_values']

        # 3. Tokenize the prompt, correctly handling the special image token
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer_, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

        return input_ids

    def preprocessor(
        self,
        instruction: str = "What are these?",
        image_file: str = "",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ):
        preprocessor = self.get_preprocessor()
        inputs = preprocessor(
            instruction=instruction,
            image_file=image_file,
            response=response,
            generation_mode=generation_mode,
        )
        return inputs

    def generate(
        self,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        **inputs: Dict[str, Any],
    ):

        # # 4. Set up stopping criteria to end generation at the right time
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer_, input_ids)
        
        #inputs = {k: v.unsqueeze(0).to(self.model_.device) for k, v in inputs.items()}
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(
            device_type=device_type, enabled=True, dtype=self.model_.dtype
        ):
            output = self.model_.generate(
                inputs,
                images=inputs['images'],
                #**generation_kwargs
            )
        return output
