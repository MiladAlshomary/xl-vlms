from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image
import sys
import os
import re

sys.path.append(os.environ['LLAVA_PATH'])

from .image_text_model import ImageTextModel

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import requests
from io import BytesIO


__all__ = ["GalleryGPT"]


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

class GalleryGPT(ImageTextModel):

    def set_model(
        self,
    ) -> None:

        # Model
        disable_torch_init()

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_name_or_path, 'Lin-Chen/ShareGPT4V-7B', 'llava-lora-model'
        )

        # Iterate through named modules and print their names
        for name, module in model.named_modules():
            print(name)
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

        return self.model_.lm_head

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
        conv_mode: str ='llava_v0',
        **kwargs: Any,
    ) -> Dict[str, Any]:

        model = self.model_.model

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in instruction:
            if model.config.mm_use_im_start_end:
                instruction = re.sub(IMAGE_PLACEHOLDER, image_token_se, instruction)
            else:
                instruction = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, instruction)
        else:
            if model.config.mm_use_im_start_end:
                instruction = image_token_se + "\n" + instruction
            else:
                instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(image_file)
        image_sizes = [image.size]
        images_tensor = process_images(
            [image],
            self.processor_,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer_, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        self._image_sizes  = image_sizes
        self._image_tensors = images_tensor
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
        return {'input_ids' : inputs}

    def generate(
        self,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        **inputs: Dict[str, Any],
    ):
        
        with torch.inference_mode():
            output_ids = self.model_.generate(
                inputs['input_ids'],
                images=self._image_tensors,
                image_sizes=self._image_sizes,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )

            #outputs = self.tokenizer_.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return output_ids