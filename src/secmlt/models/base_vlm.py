"""Basic wrapper for generic vision-language models."""

from abc import abstractmethod

import torch
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader

IGNORE_INDEX = -100
"""Padding value for target token IDs, ignored when computing the loss."""


class BaseVisionLanguageModel(BaseModel):
    """
    Abstract base class for vision-language models (VLMs).

    A VLM answers a text prompt about an input image. To be attacked with
    gradient-based evasion attacks, the model exposes the logits of a target
    answer (computed with teacher forcing) as its decision function, so that
    the gradient of any token-level loss can flow back to the input pixels.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> torch.LongTensor:
        """
        Convert target answers into token IDs.

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
        ...

    @abstractmethod
    def decode(self, ids: torch.LongTensor) -> list[str]:
        """
        Convert token IDs back into text.

        Parameters
        ----------
        ids : torch.LongTensor
            Token IDs of shape [batch, seq_len]. IGNORE_INDEX values are skipped.

        Returns
        -------
        list of str
            Decoded texts.
        """
        ...

    @abstractmethod
    def generate(self, x: torch.Tensor, prompt: str, **kwargs) -> list[str]:
        """
        Generate the answers to the prompt for each input image.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width],
            with values in [0, 1].
        prompt : str
            Text prompt (e.g., a question about the image).

        Returns
        -------
        list of str
            Generated answers, one per image.
        """
        ...

    def decision_function(
        self, x: torch.Tensor, prompt: str, target_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute the logits of the target answer given image and prompt.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width],
            with values in [0, 1].
        prompt : str
            Text prompt (e.g., a question about the image).
        target_ids : torch.LongTensor
            Target token IDs of shape [batch, target_len], as returned
            by `encode`.

        Returns
        -------
        torch.Tensor
            Logits predicting each target token, of shape
            [batch, target_len, vocab_size].
        """
        x = self._preprocessing(x)
        logits = self._decision_function(x, prompt, target_ids)
        return self._postprocessing(logits)

    @abstractmethod
    def _decision_function(
        self, x: torch.Tensor, prompt: str, target_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute the logits of the target answer given image and prompt.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width],
            with values in [0, 1].
        prompt : str
            Text prompt (e.g., a question about the image).
        target_ids : torch.LongTensor
            Target token IDs of shape [batch, target_len], as returned
            by `encode`.

        Returns
        -------
        torch.Tensor
            Logits predicting each target token, of shape
            [batch, target_len, vocab_size]. Must be differentiable
            with respect to x.
        """
        ...

    def predict(self, x: torch.Tensor, prompt: str, **kwargs) -> list[str]:
        """
        Shortcut for self.generate().

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [batch, channels, height, width].
        prompt : str
            Text prompt.

        Returns
        -------
        list of str
            Generated answers, one per image.
        """
        return self.generate(x, prompt, **kwargs)

    def train(self, dataloader: DataLoader) -> "BaseVisionLanguageModel":
        """
        Train the model (not supported).

        Parameters
        ----------
        dataloader : DataLoader
            Train data loader.

        Raises
        ------
        NotImplementedError
            Training vision-language models is not supported.
        """
        msg = "Training is not supported for vision-language models."
        raise NotImplementedError(msg)
