"""Wrapper for Hugging Face vision-language models."""

import torch
from secmlt.models.base_vlm import IGNORE_INDEX, BaseVisionLanguageModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BatchFeature


class HFVisionLanguageModel(BaseVisionLanguageModel):
    """
    Wrapper for Hugging Face image-text-to-text models (e.g., Qwen-VL).

    Input images are processed with the model's (fast) Hugging Face image
    processor, which is implemented with torch operations. Hence, the
    pixel values fed to the model remain differentiable with respect to the
    input images.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        processor_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        """
        Create a wrapped Hugging Face vision-language model.

        Parameters
        ----------
        model_path : str
            Model name or local path.
        device : torch.device, optional
            Device where the model is loaded. Defaults to GPU if available.
        dtype : torch.dtype, optional
            Model precision. Defaults to torch.float32.
        processor_kwargs : dict, optional
            Extra arguments for AutoProcessor.from_pretrained().
        model_kwargs : dict, optional
            Extra arguments for AutoModelForImageTextToText.from_pretrained().
        """
        super().__init__()
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._dtype = dtype or torch.float32
        self._processor = AutoProcessor.from_pretrained(
            model_path, **(processor_kwargs or {})
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=self._dtype, **(model_kwargs or {})
        ).eval()
        self._model.to(self._device)
        # gradients are only needed w.r.t. the input images
        self._model.requires_grad_(requires_grad=False)

    @property
    def model(self) -> AutoModelForImageTextToText:
        """
        Get the wrapped Hugging Face model.

        Returns
        -------
        AutoModelForImageTextToText
            Wrapped Hugging Face model.
        """
        return self._model

    @property
    def processor(self) -> AutoProcessor:
        """
        Get the wrapped Hugging Face processor.

        Returns
        -------
        AutoProcessor
            Wrapped Hugging Face processor.
        """
        return self._processor

    def _prepare_inputs(self, x: torch.Tensor, prompt: str) -> BatchFeature:
        """Build the model inputs from images in [0, 1] and a text prompt."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            images=list(x.to(self._device)),
            text=[text] * x.shape[0],
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        return inputs

    def encode(self, texts: list[str]) -> torch.LongTensor:
        """
        Tokenize target answers, appending the end-of-turn token.

        Parameters
        ----------
        texts : list of str
            Target answers, one per sample.

        Returns
        -------
        torch.LongTensor
            Token IDs of shape [batch, max_target_len], right-padded
            with IGNORE_INDEX.
        """
        tokenizer = self._processor.tokenizer
        ids = [
            torch.tensor(
                [
                    *tokenizer(t, add_special_tokens=False).input_ids,
                    tokenizer.eos_token_id,
                ]
            )
            for t in texts
        ]
        return torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=IGNORE_INDEX
        )

    def decode(self, ids: torch.LongTensor) -> list[str]:
        """
        Decode token IDs into text, skipping padding and special tokens.

        Parameters
        ----------
        ids : torch.LongTensor
            Token IDs of shape [batch, seq_len].

        Returns
        -------
        list of str
            Decoded texts.
        """
        return self._processor.batch_decode(
            [row[row != IGNORE_INDEX] for row in ids], skip_special_tokens=True
        )

    def _decision_function(
        self, x: torch.Tensor, prompt: str, target_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute the logits of the target answer with teacher forcing.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width],
            with values in [0, 1].
        prompt : str
            Text prompt (e.g., a question about the image).
        target_ids : torch.LongTensor
            Target token IDs of shape [batch, target_len].

        Returns
        -------
        torch.Tensor
            Logits predicting each target token, of shape
            [batch, target_len, vocab_size].
        """
        inputs = self._prepare_inputs(x, prompt)
        prompt_len = inputs["input_ids"].shape[1]
        target_len = target_ids.shape[1]
        # padding positions come after the answer, so with causal attention
        # they do not affect the logits of the actual target tokens
        target_ids = target_ids.to(self._device).clamp(min=0)

        # append the target answer to every sequence-aligned input
        for key, value in inputs.items():
            if value.ndim == 2 and value.shape[1] == prompt_len:  # noqa: PLR2004
                if key == "input_ids":
                    suffix = target_ids
                elif key == "attention_mask":
                    suffix = torch.ones_like(target_ids)
                else:  # e.g., token type ids (0 = text)
                    suffix = torch.zeros_like(target_ids)
                inputs[key] = torch.cat([value, suffix.to(value.dtype)], dim=1)

        logits = self._model(**inputs).logits
        # logits at position i predict the token at position i + 1
        return logits[:, prompt_len - 1 : prompt_len - 1 + target_len, :]

    @torch.no_grad()
    def generate(self, x: torch.Tensor, prompt: str, **kwargs) -> list[str]:
        """
        Generate the answers to the prompt (greedy decoding by default).

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width],
            with values in [0, 1].
        prompt : str
            Text prompt (e.g., a question about the image).
        **kwargs
            Additional parameters for model.generate().

        Returns
        -------
        list of str
            Generated answers, one per image.
        """
        inputs = self._prepare_inputs(x, prompt)
        kwargs.setdefault("max_new_tokens", 32)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("eos_token_id", self._processor.tokenizer.eos_token_id)
        kwargs.setdefault("pad_token_id", self._processor.tokenizer.pad_token_id)
        out = self._model.generate(**inputs, **kwargs)
        # answers are not stripped, so that encode(generate(...)) gives back
        # the generated tokens (e.g., " Black" and "Black" are different tokens)
        return self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
