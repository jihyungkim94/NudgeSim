"""Regenerate the README overview figures.

    python scripts/make_overview_figures.py

v4 reads runs/main/, so it reflects whatever run is currently committed.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMGS = os.path.join(REPO, "imgs")

INK = "#1A1A1A"
MUTED = "#5A5A5A"
LEFT = "#2E6FB7"
RIGHT = "#C2453C"
NEUTRAL = "#7A7A7A"
BAD = "#2B2B2B"
AI = "#1F8A70"
LIGHT = "#F4F6F9"
EDGE = "#C9D2DD"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, fc=LIGHT, ec=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=fc, ec=ec, lw=lw, zorder=1))


def arrow(ax, a, b, color, lw=2.0, rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=14, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- version 1
def v1_society():
    fig = plt.figure(figsize=(10, 4.8), dpi=200)
    fig.text(0.5, 0.965, "NudgeSim: who pays to correct?", fontsize=16,
             fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.902,
             "Seven agents, one false claim, twelve rounds. Correcting a peer costs you \u03ba \u2014 so who does it?",
             fontsize=9.5, color=MUTED, ha="center", va="top")

    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.05,
                          left=0.02, right=0.98, top=0.85, bottom=0.02)

    # --- network, equal aspect so nodes are circles
    axn = fig.add_subplot(gs[0, 0])
    axn.set_aspect("equal"); axn.axis("off")
    axn.set_xlim(0, 1); axn.set_ylim(0, 1)

    D = (0.30, 0.86)
    A1, A2 = (0.12, 0.52), (0.34, 0.46)
    B1, B2 = (0.62, 0.54), (0.72, 0.80)
    C = (0.44, 0.20)
    DA = (0.54, 0.93)

    for a, b in [(D, A1), (D, A2), (D, B1), (D, B2), (A1, A2), (A2, C),
                 (B1, C), (A2, B1), (B1, B2)]:
        axn.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=1.6, zorder=1)
    for a in (B1, B2, A2):
        axn.plot([DA[0], a[0]], [DA[1], a[1]], color="#BFE3D9", lw=2.0, zorder=1)

    def node(xy, label, color, r=0.062):
        axn.add_patch(plt.Circle(xy, r, color=color, zorder=3))
        axn.text(xy[0], xy[1], label, ha="center", va="center", fontsize=9,
                 color="white", fontweight="bold", zorder=4)

    node(D, "DIS", BAD); node(DA, "DA", AI)
    node(A1, "A1", LEFT); node(A2, "A2", LEFT)
    node(B1, "B1", RIGHT); node(B2, "B2", RIGHT); node(C, "C", NEUTRAL)

    axn.annotate("Disseminator\ninjects a LIAR claim", xy=(D[0] - 0.05, D[1] + 0.03),
                 xytext=(0.02, 0.97), fontsize=8, color=BAD, ha="left", va="top",
                 linespacing=1.4,
                 arrowprops=dict(arrowstyle="-", color=BAD, lw=0.9, shrinkA=0, shrinkB=4))
    axn.annotate("Devil's Advocate\nthe AI intervener", xy=(DA[0] + 0.05, DA[1] - 0.02),
                 xytext=(0.70, 0.99), fontsize=8, color=AI, ha="left", va="top",
                 linespacing=1.4,
                 arrowprops=dict(arrowstyle="-", color=AI, lw=0.9, shrinkA=0, shrinkB=4))
    axn.text(0.5, 0.035, "wired from real PHEME reply trees", fontsize=8.2,
             color=MUTED, ha="center", style="italic")

    # --- right column
    axr = fig.add_subplot(gs[0, 1]); blank(axr)

    box(axr, 0.02, 0.50, 0.96, 0.46)
    axr.text(0.06, 0.905, "Each round, every citizen picks one:",
             fontsize=9.5, fontweight="bold", va="top")
    for i, (act, note, col) in enumerate([
            ("SHARE", "spread it", RIGHT),
            ("ENDORSE", "back it", RIGHT),
            ("CHALLENGE", "pay \u03ba to correct it", AI),
            ("IGNORE", "stay out", MUTED)]):
        y = 0.815 - i * 0.072
        axr.text(0.08, y, act, fontsize=9.5, fontweight="bold", color=col, va="center")
        axr.text(0.42, y, note, fontsize=8.5, color=MUTED, va="center")
    axr.text(0.06, 0.545,
             r"$\pi = \beta C + \gamma V - \kappa\,\mathbf{1}[\mathrm{CHALLENGE}] - \delta D$",
             fontsize=10.5, va="center")

    box(axr, 0.02, 0.05, 0.96, 0.38, fc="#EAF1F8", ec="#BBD0E6")
    axr.text(0.06, 0.385, "Read straight off the action log:", fontsize=9.5,
             fontweight="bold", va="top")
    axr.text(0.08, 0.285, "FPR", fontsize=10.5, fontweight="bold", color=RIGHT, va="center")
    axr.text(0.24, 0.285, "how far the false claim spreads", fontsize=8.5,
             color=MUTED, va="center")
    axr.text(0.08, 0.185, "EPC", fontsize=10.5, fontweight="bold", color=AI, va="center")
    axr.text(0.24, 0.185, "citizens who pay \u03ba themselves", fontsize=8.5,
             color=MUTED, va="center")
    axr.text(0.08, 0.105, "the headline: second-order cooperation", fontsize=8.2,
             color=AI, style="italic", va="center")

    fig.savefig(os.path.join(IMGS, "nudgesim_society.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- version 2
def v2_question():
    fig, ax = plt.subplots(figsize=(9.6, 4.5), dpi=200)
    blank(ax)

    ax.text(0.5, 0.985, "Does an AI corrector build the norm, or replace it?",
            fontsize=15.5, fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.915,
            "The intervener is stipulated to pay the correction cost. The question is what the citizens then do.",
            fontsize=9.5, color=MUTED, ha="center", va="top")

    box(ax, 0.02, 0.245, 0.235, 0.50)
    ax.text(0.1375, 0.715, "The dilemma", fontsize=10.5, fontweight="bold",
            ha="center", va="top")
    ax.text(0.1375, 0.625,
            "Accurate information\nis a public good.\n\nVerifying costs you \u03ba;\n"
            "a cleaner feed\nbenefits everyone.",
            fontsize=8.8, color=MUTED, ha="center", va="top", linespacing=1.6)
    ax.text(0.1375, 0.275, "\u2192 nobody wants to pay", fontsize=8.6, color=RIGHT,
            ha="center", va="center", style="italic")

    arrow(ax, (0.268, 0.495), (0.335, 0.495), MUTED, lw=1.8)
    # figure is 9.6 x 4.5, so widen in x to render a visual circle
    ax.add_patch(mpatches.Ellipse((0.40, 0.495), width=0.092 * (4.5 / 9.6) * 2,
                                  height=0.092, color=AI, zorder=3))
    ax.text(0.40, 0.495, "DA", ha="center", va="center", fontsize=10,
            color="white", fontweight="bold", zorder=4)
    ax.text(0.40, 0.415, "enters early or late,\nnudging or debunking",
            fontsize=8.5, color=AI, ha="center", va="top", linespacing=1.5)

    arrow(ax, (0.452, 0.545), (0.595, 0.695), AI, lw=2.4, rad=0.16)
    arrow(ax, (0.452, 0.445), (0.595, 0.295), RIGHT, lw=2.4, rad=-0.16)

    box(ax, 0.605, 0.575, 0.375, 0.235, fc="#E8F5F1", ec="#A9D8CA")
    ax.text(0.63, 0.775, "CATALYSE", fontsize=11.5, fontweight="bold", color=AI, va="top")
    ax.text(0.63, 0.700,
            "citizens start paying \u03ba themselves \u2014\nthe norm outlives the intervention",
            fontsize=8.8, color=MUTED, va="top", linespacing=1.55)

    box(ax, 0.605, 0.175, 0.375, 0.235, fc="#FBEEED", ec="#EEC4C0")
    ax.text(0.63, 0.375, "CROWD OUT", fontsize=11.5, fontweight="bold", color=RIGHT, va="top")
    ax.text(0.63, 0.300,
            "citizens leave it to the corrector \u2014\npropagation falls, the norm erodes",
            fontsize=8.8, color=MUTED, va="top", linespacing=1.55)

    ax.text(0.5, 0.065,
            "Both look like success if you only measure propagation. NudgeSim measures what agents pay for.",
            fontsize=9, ha="center", style="italic")

    fig.savefig(os.path.join(IMGS, "nudgesim_question.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- version 3
def v3_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    blank(ax)
    ax.set_xlim(-0.035, 1.035); ax.set_ylim(-0.04, 1.04)

    ax.text(0.5, 0.985, "NudgeSim: a payoff-instrumented testbed for LLM agent societies",
            fontsize=14.5, fontweight="bold", ha="center", va="top")

    stages = [
        ("Real corpora", "LIAR claims\nPHEME topologies", "#EAF1F8", "#BBD0E6"),
        ("Society", "5 citizens + bad actor\n+ AI intervener", LIGHT, EDGE),
        ("Design", "timing \u00d7 tone\n\u00d7 6 backbones", LIGHT, EDGE),
        ("Outcomes", "FPR \u00b7 EPC\nread off the log", "#E8F5F1", "#A9D8CA"),
    ]
    w, gap = 0.212, 0.042
    x0 = (1.0 - (4 * w + 3 * gap)) / 2
    for i, (title, body, fc, ec) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(ax, x, 0.565, w, 0.29, fc=fc, ec=ec)
        ax.text(x + w / 2, 0.815, title, fontsize=10.5, fontweight="bold",
                ha="center", va="top")
        ax.text(x + w / 2, 0.728, body, fontsize=8.8, color=MUTED, ha="center",
                va="top", linespacing=1.6)
        if i < 3:
            arrow(ax, (x + w + 0.004, 0.71), (x + w + gap - 0.004, 0.71), MUTED, lw=1.8)

    box(ax, x0, 0.075, 4 * w + 3 * gap, 0.41, fc="white")
    ax.text(x0 + 0.025, 0.445, "Backbones run as monocultures \u2014 and as one mixed society",
            fontsize=10.5, fontweight="bold", va="top")
    ax.text(x0 + 0.025, 0.372,
            "Seating the whole roster in a single episode, rotating which model holds which seat, is what lets the design ask\n"
            "whether one model free-rides on another's correction. A one-model-per-episode grid cannot pose that question.",
            fontsize=8.8, color=MUTED, va="top", linespacing=1.65)

    chips = [("GPT-4o", LEFT), ("OpenAI reasoning", LEFT), ("Opus 5", AI),
             ("Opus 5 + thinking", AI), ("Llama-3.3-70B", NEUTRAL), ("Qwen3-30B", NEUTRAL)]
    cw, cgap = 0.138, 0.018
    cx0 = x0 + 0.025
    for i, (lbl, col) in enumerate(chips):
        x = cx0 + i * (cw + cgap)
        ax.add_patch(FancyBboxPatch((x, 0.135), cw, 0.062,
                                    boxstyle="round,pad=0.008", fc=LIGHT, ec=col, lw=1.3))
        ax.text(x + cw / 2, 0.166, lbl, fontsize=7.4, color=col, ha="center",
                va="center", fontweight="bold")

    fig.savefig(os.path.join(IMGS, "nudgesim_pipeline.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- version 4
def v4_measured():
    rounds = pd.read_parquet(REPO + "/runs/main/rounds.parquet")
    core = rounds[rounds["arm"] == "core"]
    traj = core.groupby(["timing", "round"])["fpr"].mean().reset_index()

    cm = pd.read_csv(REPO + "/runs/main/analysis/cell_means.csv")
    epc = cm.groupby(["timing", "tone"])["epc_mean"].mean()

    fig = plt.figure(figsize=(10, 4.5), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.3,
                          left=0.07, right=0.975, top=0.78, bottom=0.16)

    fig.text(0.5, 0.975, "Intervene early or late, nudge or debunk \u2014 and watch who pays \u03ba",
             fontsize=14, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.905,
             "Outcomes are actions in the log, not beliefs the agents report.",
             fontsize=9, color=MUTED, ha="center", va="top")

    ax = fig.add_subplot(gs[0, 0])
    styles = {"none": (MUTED, (0, (4, 2)), "no intervener"),
              "early": (AI, "-", "early entry (round 2)"),
              "late": (LEFT, "-", "late entry (round 6)")}
    for key, (col, ls, lab) in styles.items():
        sub = traj[traj["timing"] == key]
        ax.plot(sub["round"], sub["fpr"], color=col, lw=2.5, linestyle=ls, label=lab)
    ax.axvline(2, color=AI, lw=1.0, alpha=0.4)
    ax.axvline(6, color=LEFT, lw=1.0, alpha=0.4)
    ax.set_xlabel("round", fontsize=9)
    ax.set_ylabel("false-claim propagation (FPR)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    bars = [("control", epc[("none", "none")], NEUTRAL),
            ("aggressive\ndebunking", epc[("early", "aggressive_debunking")], RIGHT),
            ("empathetic\nnudge", epc[("early", "empathetic_nudge")], AI)]
    ax2.bar([b[0] for b in bars], [b[1] for b in bars],
            color=[b[2] for b in bars], width=0.58)
    for i, b in enumerate(bars):
        ax2.text(i, b[1] + 0.004, f"{b[1]:.3f}", ha="center", fontsize=8.5, color=INK)
    ax2.set_ylabel("citizens paying \u03ba themselves (EPC)", fontsize=9)
    ax2.set_ylim(0, max(b[1] for b in bars) * 1.30)
    ax2.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.text(0.5, 1.02, "the headline outcome, early entry", transform=ax2.transAxes,
             fontsize=8.5, ha="center", style="italic")

    fig.text(0.5, 0.035,
             "Measured on the analytic surrogate policy over real LIAR claims and PHEME topologies \u2014 "
             "a property of the declared process, not a result about language models.",
             fontsize=7.8, color=MUTED, ha="center", style="italic")

    fig.savefig(os.path.join(IMGS, "nudgesim_measured.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(IMGS, exist_ok=True)
    v1_society(); v2_question(); v3_pipeline(); v4_measured()
    print("done")
