import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list, temperature: float, top_p: float) -> str:
        pass

class ChatLLM(BaseLLM):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map='auto',
            # attn_implementation='flash_attention_2',
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model.eval()


    def generate(self, messages, temperature=0.0, top_p=1.0) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}


            gen_kwargs = {
                "max_new_tokens": 448,
            }

            if temperature is not None and temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p) if top_p is not None else 1.0
            else:
                gen_kwargs["do_sample"] = False
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
                # # Debug
                # gen_len = generated_ids.shape[1] - inputs["input_ids"].shape[1]
                # print(f"[DEBUG] gen_len={gen_len}, max_new_tokens={gen_kwargs['max_new_tokens']}")

            response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
            response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            raise RuntimeError(f"Generate failed for model {self.model_id}: {e}") from e
