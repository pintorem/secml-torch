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

    The base class handles targeted and untargeted attacks once: subclasses
    only define how to compute the scores, the per-sample loss and whether
    the prediction matches a reference (the labels, or the target).
    """

    def __init__(self, y_target: int | torch.Tensor | None = None) -> None:
        """
        Create the objective.

        Parameters
        ----------
        y_target : int | torch.Tensor | None, optional
            Target for targeted attacks, None for untargeted attacks.
            By default None.
        """
        self.y_target = y_target

    def reference(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get the reference of the loss: the labels, or the target.

        Parameters
        ----------
        labels : torch.Tensor
            Labels of the samples, as given by the data loader.

        Returns
        -------
        torch.Tensor
            The labels (untargeted) or the target for each sample (targeted).
        """
        if self.y_target is None:
            return labels
        return (torch.zeros_like(labels) + self.y_target).type(labels.dtype)

    @abstractmethod
    def scores(
        self, model: BaseModel, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the model scores.

        Parameters
        ----------
        model : BaseModel
            Attacked model.
        x : torch.Tensor
            Input samples (differentiable w.r.t. the perturbation).
        reference : torch.Tensor
            The labels (untargeted) or the target (targeted).

        Returns
        -------
        torch.Tensor
            Model scores.
        """
        ...

    @abstractmethod
    def loss(self, scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-sample loss w.r.t. the reference.

        Parameters
        ----------
        scores : torch.Tensor
            Model scores.
        reference : torch.Tensor
            The labels (untargeted) or the target (targeted).

        Returns
        -------
        torch.Tensor
            Per-sample loss of shape [batch].
        """
        ...

    @abstractmethod
    def matches(self, scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Check whether the model prediction matches the reference.

        Parameters
        ----------
        scores : torch.Tensor
            Model scores.
        reference : torch.Tensor
            The labels (untargeted) or the target (targeted).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [batch].
        """
        ...

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
            Model scores and per-sample objective to maximize, i.e., the loss
            (untargeted) or the negated loss (targeted).
        """
        multiplier = 1 if self.y_target is None else -1
        reference = self.reference(labels)
        scores = self.scores(model, x, reference)
        return scores, self.loss(scores, reference) * multiplier

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
            Boolean tensor of shape [batch]: the prediction differs from the
            labels (untargeted) or matches the target (targeted).
        """
        matches = self.matches(scores, self.reference(labels))
        if self.y_target is None:
            return ~matches
        return matches


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
        super().__init__(y_target=y_target)
        self.loss_function = loss_function

    def scores(
        self, model: BaseModel, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the output scores of the classifier.

        Parameters
        ----------
        model : BaseModel
            Attacked classifier.
        x : torch.Tensor
            Input samples.
        reference : torch.Tensor
            The labels or the target class (not used by the forward pass).

        Returns
        -------
        torch.Tensor
            Output scores of shape [batch, classes].
        """
        return model.decision_function(x)

    def loss(self, scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-sample classification loss.

        Parameters
        ----------
        scores : torch.Tensor
            Output scores.
        reference : torch.Tensor
            The labels or the target class.

        Returns
        -------
        torch.Tensor
            Per-sample loss.
        """
        return self.loss_function(scores, reference.to(scores.device))

    def matches(self, scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Check whether the predicted class is the reference class.

        Parameters
        ----------
        scores : torch.Tensor
            Output scores.
        reference : torch.Tensor
            The labels or the target class.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [batch].
        """
        return scores.argmax(dim=1) == reference
