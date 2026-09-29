import json

import matplotlib.pyplot as plt


def read_validation_loss(metrics_path):
    steps = []
    validation_losses = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if "val_loss" in record:
                # 日志中的 step 从 0 开始
                steps.append(record["step"] + 1)
                validation_losses.append(record["val_loss"])

    return steps, validation_losses


steps, validation_losses = read_validation_loss(
    "result/metrics.jsonl"
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    steps,
    validation_losses,
    marker="o",
    markersize=4,
    linewidth=2,
    color="tab:orange",
)

ax.set_title("Validation Loss Curve")
ax.set_xlabel("Training Step")
ax.set_ylabel("Validation Loss")
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    "result/validation_loss_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()