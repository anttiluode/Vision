import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/eraser_receipt.json"))
S, C = R["summary_mean_sd"], R["setup"]["delays_cycles"]
none = S["none|0"]["phase_damage"][0]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.axvspan(0, R["setup"]["listen_time"] / R["setup"]["T_cycle"], color="#eeeeee", label="listening window")
ax.axhline(none, color="k", ls="--", lw=1.2, label=f"no eraser ({none:.3f})")
for er, col, lab in (("input_locked", "#2a6fdb", "eraser carries a copy of the INPUT"),
                     ("rate_driven", "#d9822b", "eraser recruited by the neuron's firing RATE")):
    y = [S[f"{er}|{c:g}"]["phase_damage"][0] for c in C]
    ax.plot(C, y, "o-", color=col, label=lab)
ax.set_yscale("log")
ax.set_xlabel("eraser delay after the query (oscillation cycles)")
ax.set_ylabel("lasting phase damage to the tuft memory (rad)")
ax.set_title("Delayed tuft inhibition erases a read only on whole cycles, and only with a copy of the input", fontsize=10)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout(); fig.savefig("results/eraser_summary.png", dpi=130)
