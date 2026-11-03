"""
Targeted attack against a vision-language model (Qwen3.5 / Qwen3.8).

Perturb an image (by default, a black image) so that the model answers the
prompt with the given target answer, using PGD (fixed Linf budget) or DDN
(minimum L2 perturbation). The attacks are the standard native ones: only the
objective changes w.r.t. classifiers.
The loss and the perturbation norms are tracked at each step, and saved in a
plot (vlm_attack_<attack>.png).

Usage (requires `pip install secml-torch[llm]` with transformers>=5.2):

    python examples/vlm_targeted_attack.py --attack pgd
    python examples/vlm_targeted_attack.py --attack ddn

The default model (Qwen/Qwen3.5-0.8B) runs on CPU. The same wrapper works
with larger models of the same family, e.g., Qwen/Qwen3.8-27B, on GPU.
"""

import argparse

import matplotlib.pyplot as plt
import torch
from PIL import Image
from secmlt.adv import Backends
from secmlt.adv.evasion import DDN, PGD, LpPerturbationModels
from secmlt.models.hugging_face import HFVisionLanguageModel
from secmlt.optimization import TeacherForcingObjective
from secmlt.trackers import LossTracker, PerturbationNormTracker
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import pil_to_tensor

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--attack", choices=["pgd", "ddn"], default="pgd")
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--image", default=None, help="Input image (default: black)")
parser.add_argument("--size", type=int, default=256, help="Side of the image")
parser.add_argument("--prompt", default="What color is this image? One word.")
parser.add_argument("--target", default=" white", help="Target answer")
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()

model = HFVisionLanguageModel(args.model)

# a [1, 3, H, W] image in [0, 1]. A side multiple of 32 avoids the processor
# resizing the image (and the perturbation)
if args.image is None:
    image = torch.zeros(1, 3, args.size, args.size)
else:
    pil = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    image = pil_to_tensor(pil).unsqueeze(0).float() / 255

# as for classifiers, the labels are the original (clean) answers, and the
# target answer is given as y_target
clean_answer = model.generate(image, args.prompt)
loader = DataLoader(TensorDataset(image, model.encode(clean_answer)))
objective = TeacherForcingObjective(args.prompt, y_target=model.encode([args.target]))

# trackers of the loss and of the perturbation norms at each step
trackers = {
    "loss": LossTracker(),
    "linf": PerturbationNormTracker(LpPerturbationModels.LINF),
    "l2": PerturbationNormTracker(LpPerturbationModels.L2),
}
attacks = {
    "pgd": lambda: PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=32 / 255,
        num_steps=args.steps,
        step_size=2 / 255,
        objective=objective,
        backend=Backends.NATIVE,
        trackers=list(trackers.values()),
    ),
    "ddn": lambda: DDN(
        num_steps=args.steps,
        eps_init=4.0,
        objective=objective,
        backend=Backends.NATIVE,
        trackers=list(trackers.values()),
    ),
}
adv_image, _ = next(iter(attacks[args.attack]()(model, loader)))

# the tracked loss is the objective (maximized): for targeted attacks it is the
# negated cross-entropy of the target answer
target_ce = -trackers["loss"].get()[0]
linf = trackers["linf"].get()[0] * 255
l2 = trackers["l2"].get()[0]
for step in range(0, len(target_ce), 10):
    print(
        f"step {step:3d}  target CE={target_ce[step]:.3f}  "
        f"Linf={linf[step]:.1f}/255  L2={l2[step]:.2f}"
    )

delta = adv_image - image
adv_answer = model.generate(adv_image, args.prompt)[0]
print(f"Prompt:             {args.prompt!r}")
print(f"Clean answer:       {clean_answer[0]!r}")
print(f"Target answer:      {args.target!r}")
print(f"Adversarial answer: {adv_answer!r}")
print(
    f"{args.attack.upper()} perturbation: Linf={delta.abs().max() * 255:.1f}/255, "
    f"L2={delta.norm():.2f}"
)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(target_ce)
ax[0].set(title="Cross-entropy of the target answer", xlabel="step")
ax[1].plot(linf, label="Linf (x255)")
ax[1].plot(l2, label="L2")
ax[1].set(title="Perturbation norm", xlabel="step")
ax[1].legend()
amplified = (0.5 + 4 * delta[0]).clamp(0, 1)
ax[2].imshow(torch.cat([adv_image[0], amplified], dim=2).permute(1, 2, 0))
ax[2].set(title="Adversarial image | perturbation (x4)")
ax[2].axis("off")
fig.suptitle(
    f"{args.attack.upper()}: clean {clean_answer[0][:20]!r}, "
    f"target {args.target[:20]!r}, adversarial {adv_answer[:20]!r}"
)
fig.tight_layout()
fig.savefig(f"vlm_attack_{args.attack}.png", dpi=120)
print(f"Saved vlm_attack_{args.attack}.png")
