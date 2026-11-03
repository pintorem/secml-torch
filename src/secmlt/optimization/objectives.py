"""Objectives optimized by the modular attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from secmlt.models import BaseModel


class Objective(ABC):
    """
    Objective of an attack: what to optimize and when the attack succeeds.

    It decouples the attack loops from the kind of attacked model. The loops
    only maximize the per-sample objective, and use the success criterion
    for minimum-distance attacks.
    """

    @abstractmethod
    def __call__(
        self, model: BaseModel, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the model scores and the objective to maximize.

        Parameters
        ----------
        model : BaseModel
            Attacked model.
        x : torch.Tensor
            Input samples (differentiable w.r.t. the perturbation).
        labels : torch.Tensor
            Labels of the samples, as given by the data loader.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Model scores and per-sample objective to maximize.
        """
        ...

    @abstractmethod
    def is_success(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Check whether the attack succeeded on each sample.

        Parameters
        ----------
        scores : torch.Tensor
            Model scores.
        labels : torch.Tensor
            Labels of the samples, as given by the data loader.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [batch].
        """
        ...


class ClassificationObjective(Objective):
    """
    Objective for classifiers.

    Untargeted attacks maximize the loss of the true labels, targeted attacks
    minimize the loss of the target class.
    """

    def __init__(
        self, loss_function: torch.nn.Module, y_target: int | None = None
    ) -> None:
        """
        Create the classification objective.

        Parameters
        ----------
        loss_function : torch.nn.Module
            Per-sample classification loss.
        y_target : int | None, optional
            Target class for targeted attacks, None for untargeted attacks.
            By default None.
        """
        self.loss_function = loss_function
        self.y_target = y_target

    def _target(self, labels: torch.Tensor) -> torch.Tensor:
        """Get the true labels (untargeted) or the target class (targeted)."""
        if self.y_target is None:
            return labels
        return (torch.zeros_like(labels) + self.y_target).type(labels.dtype)

    def __call__(
        self, model: BaseModel, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scores and the signed loss.

        Parameters
        ----------
        model : BaseModel
            Attacked classifier.
        x : torch.Tensor
            Input samples.
        labels : torch.Tensor
            True labels of the samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output scores and per-sample loss, negated for targeted attacks.
        """
        multiplier = 1 if self.y_target is None else -1
        scores = model.decision_function(x)
        target = self._target(labels).to(scores.device)
        losses = self.loss_function(scores, target)
        return scores, losses * multiplier

    def is_success(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Check misclassification (untargeted) or target-class prediction (targeted).

        Parameters
        ----------
        scores : torch.Tensor
            Output scores.
        labels : torch.Tensor
            True labels of the samples.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [batch].
        """
        target = self._target(labels)
        if self.y_target is None:
            return scores.argmax(dim=1) != target
        return scores.argmax(dim=1) == target
