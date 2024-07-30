#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#               This file was automatically generated from <path_to_diff_file.py>.
#         Do NOT edit this file manually as any edits will be overwritten by the generation of
#         the file from the diff. If any change should be done, please apply the change to the
#                                    diff.py file directly.
#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

class CostWiseGemmaConfig(Gemma2Config):
    r"""
    This is the configuration class to store the configuration of a [`GemmaModel`]. It is used to instantiate an Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.
    e.g. [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        start_layer (`int`, *optional*, defaults to 28):
            The start layer to output score.
        layer_sep (`int`, *optional*, defaults to 28):
            The sep layer from the start layer to output score.
        layer_wise (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be layerwise.
    ```python
    >>> from transformers import Gemma2Model, Gemma2Config
    >>> # Initializing a Gemma2 gemma2-9b style configuration
    >>> configuration = Gemma2Config()
    >>> # Initializing a model from the gemma2-9b style configuration
    >>> model = Gemma2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cost_wise_gemma"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            start_layer: int = 28,
            layer_sep: int = 28,
            layer_wise: bool = False,
            **kwargs,
    ):
        self.start_layer = start_layer
        self.layer_sep = layer_sep
        self.layer_wise = layer_wise

        super().__init__(
            **kwargs,
        )