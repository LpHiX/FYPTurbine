"""
proto_cantilever_fig.py — PROTOTYPE (not wired into the thesis build).

Three-panel borderless terminology figure for the radial-inflow cantilever
turbine, matching Martin's hand sketch:

  (L) leading-edge detail: a single MoC blade with a bubble round the inlet
      nose and the rounded-LE radius r_LE called out.
  (M) rotor inlet (3) and exit (4) velocity triangles, sketch orientation
      (tangential u vertical, meridional c_m horizontal), full component
      decomposition. Numbers are the real design-point turbine.
  (R) the annular cascade: inner / mean / outer circles with N MoC blades,
      same silhouette as panel (L), tiled at the true pitch.

Run:  python proto_cantilever_fig.py   ->  proto_cantilever_fig.png

Once Martin signs off this gets ported into thesis_figures.py as a fig_* fn.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# thesis_figures sets up sys.path, stubs rocketcea, and exposes the cached
# design objects + the MoC blade. Single source of geometry.
import thesis_figures as tf

# ----------------------------------------------------------------------------- #
# knobs
# ----------------------------------------------------------------------------- #
N_BLADES   = 50
D_MEAN_MM  = tf.TRB["D_MEAN_MM"]          # 95 mm
SOLIDITY   = 0.92                          # blade tangential width / pitch (<1 -> gaps)
BLADE_FILL = "0.85"


# ----------------------------------------------------------------------------- #
# blade silhouette (single solid MoC blade, normalised MoC coords)
#   replicates the filled polygon built in thesis_figures._draw_passage.
#   local x = pitchwise (tangential), local y = throughflow (radial); LE nose
#   is at max-y, TE tails at min-y. Returns the nose radius too.
# ----------------------------------------------------------------------------- #
def blade_polygon(moc):
    res = moc.results
    xl, yl = moc.coords["lower_rot"]["x"], moc.coords["lower_rot"]["y"]
    xu, yu = moc.coords["upper_rot"]["x"], moc.coords["upper_rot"]["y"]
    Rl, Ru = res["Rl"], res["Ru"]
    a_l, a_u = res["alpha_lower_inlet"], res["alpha_upper_inlet"]
    y_te = yu[-1] + (xl[-1] - xu[-1]) * np.tan(moc.beta_inlet)

    th_l = np.linspace(np.pi / 2 + a_l, np.pi / 2 - a_l, 100)
    xb = np.concatenate([np.flip(xl), Rl * np.cos(th_l), -xl])
    yb = np.concatenate([np.flip(yl), Rl * np.sin(th_l), yl]) - yl[-1] + y_te

    th_u = np.linspace(np.pi / 2 + a_u, np.pi / 2 - a_u, 100)
    x_up = np.concatenate([[xl[-1]], np.flip(xu), Ru * np.cos(th_u), -xu, [-xl[-1]]])
    y_up = np.concatenate([[y_te], np.flip(yu), Ru * np.sin(th_u), yu, [y_te]])

    bx = np.concatenate([x_up, np.flip(xb)])
    by = np.concatenate([y_up, np.flip(yb)])
    return bx, by, Ru


# ----------------------------------------------------------------------------- #
# panel L — leading-edge detail
# ----------------------------------------------------------------------------- #
def panel_le(ax, bx, by, Ru):
    ax.fill(bx, by, color=BLADE_FILL, ec="k", lw=1.4, zorder=1)
    nose_x = bx[np.argmax(by)]
    nose_y = by.max()
    arc_c = (nose_x, nose_y - Ru)                       # centre of the LE arc

    # zoom bubble round the inlet nose
    bubble = plt.Circle((nose_x, nose_y - 0.45 * Ru), 1.7 * Ru,
                        fill=False, ec="0.35", lw=1.1, zorder=3)
    ax.add_patch(bubble)
    # rounded-LE radius callout: arc centre -> nose surface
    ax.annotate("", xy=(arc_c[0] + Ru * np.sin(np.deg2rad(35)),
                        arc_c[1] + Ru * np.cos(np.deg2rad(35))),
                xytext=arc_c,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="k"))
    ax.plot(*arc_c, marker=".", color="k", ms=4)
    ax.text(arc_c[0] + 0.55 * Ru, arc_c[1] + 0.55 * Ru,
            r"$r_{\mathrm{LE}}$", fontsize=12, va="bottom", ha="left")

    ax.text(nose_x, by.min() - 0.07 * (by.max() - by.min()),
            "leading-edge detail", fontsize=8, ha="center", va="top")
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
# panel M — velocity triangles (u vertical-down, c_m horizontal)
# ----------------------------------------------------------------------------- #
def _vec(ax, o, tip, lab, color, lw=1.6, off=(0, 0), ha="center", va="center"):
    ax.annotate("", xy=tip, xytext=o,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))
    mx, my = (o[0] + tip[0]) / 2, (o[1] + tip[1]) / 2
    ax.text(mx + off[0], my + off[1], lab, color=color, fontsize=11,
            ha=ha, va=va, bbox=dict(fc="white", ec="none", pad=0.2))


def _angle(ax, vertex, a_from, a_to, lab, color, r=16, lr=2.0):
    th = np.linspace(a_from, a_to, 40)
    ax.plot(vertex[0] + r * np.cos(th), vertex[1] + r * np.sin(th),
            color=color, lw=0.9)
    am = (a_from + a_to) / 2
    ax.text(vertex[0] + lr * r * np.cos(am), vertex[1] + lr * r * np.sin(am),
            lab, color=color, fontsize=11, ha="center", va="center")


def _station(ax, ox, u, cu, cm, tag, sign_labels):
    """Draw one velocity triangle at horizontal offset ox.
    cu = tangential (swirl) component (signed, + = same sense as u).
    Convention: tangential drawn DOWNWARD = -y. O at top."""
    O = np.array([ox, 0.0])
    Au = O + np.array([0.0, -u])                 # u tip
    Cu = O + np.array([0.0, -cu])                # swirl tip on the axis
    C = O + np.array([cm, -cu])                  # absolute-velocity tip
    cU, cC, cW, cK = "0.2", "C0", "C3", "0.5"

    # decomposition along the swirl axis + meridional base
    ax.plot([ox, ox], [0, -cu], color=cK, lw=0.8)            # swirl axis
    ax.plot([ox, ox + cm], [-cu, -cu], color=cK, lw=0.8)     # meridional base
    # vectors
    _vec(ax, O, Au, r"$u_%s$" % tag, cU, off=(-4, 0), ha="right")
    _vec(ax, Au, C, r"$w_%s$" % tag, cW,
         off=(4, 0) if cm > 0 else (-4, 0), ha="left" if cm > 0 else "right")
    _vec(ax, O, C, r"$c_%s$" % tag, cC, off=(5, 0), ha="left")

    # component callouts
    ax.text(ox + cm / 2, -cu - 7 * np.sign(cu or 1), r"$c_{%s m}$" % tag,
            color=cK, fontsize=10, ha="center",
            va="top" if cu > 0 else "bottom")
    ax.text(ox - 5, -cu / 2, sign_labels["cu"], color=cK, fontsize=10,
            ha="right", va="center")
    # w-swirl bracket between u tip and swirl tip (offset off the axis)
    xb = ox - 0.10 * abs(cu or u)
    ax.annotate("", xy=(xb, -cu), xytext=(xb, -u),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=cK))
    ax.text(xb - 6, -(u + cu) / 2, sign_labels["wu"], color=cK, fontsize=10,
            ha="right", va="center")

    # angle beta at u-tip (between downward tangential and w)
    a_down = -np.pi / 2
    a_w = np.arctan2(C[1] - Au[1], C[0] - Au[0])
    _angle(ax, Au, a_down, a_w, r"$\beta_%s$" % tag, cW, r=18,
           lr=1.9)
    ax.text(ox + cm / 2, min(-u, -cu, 0) - 55, "rotor %s (%s)" %
            ("inlet" if tag == "3" else "exit", tag), fontsize=8, ha="center")
    return C


def panel_triangles(ax, t):
    u, c3u, c3m = float(t.u), float(t.c3u), float(t.c3m)
    c4u, c4m = float(t.c4u), float(t.c3m)
    gap = 1.5 * max(c3u, abs(c4u), u)

    _station(ax, 0.0, u, c3u, c3m, "3",
             {"cu": r"$c_{3u}$", "wu": r"$w_{3u}$"})
    _station(ax, gap, u, c4u, c4m, "4",
             {"cu": r"$-c_{4u}$", "wu": r"$-w_{4u}$"})
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
# panel R — annular cascade (tiled at the true pitch)
# ----------------------------------------------------------------------------- #
def panel_ring(ax, bx, by):
    r_mean = D_MEAN_MM / 2.0
    pitch = 2 * np.pi * r_mean / N_BLADES
    cx, cy = bx.mean(), by.mean()
    w = bx.max() - bx.min()
    s = SOLIDITY * pitch / w                      # scale so width = solidity*pitch
    lx = (bx - cx) * s
    ly = (by - cy) * s
    half_chord = (by.max() - by.min()) * s / 2.0
    r_in, r_out = r_mean - half_chord, r_mean + half_chord

    for k in range(N_BLADES):
        th = 2 * np.pi * k / N_BLADES + np.pi / 2     # start at top
        er = np.array([np.cos(th), np.sin(th)])
        et = np.array([-np.sin(th), np.cos(th)])
        gx = lx * et[0] + (r_mean + ly) * er[0]
        gy = lx * et[1] + (r_mean + ly) * er[1]
        ax.fill(gx, gy, color=BLADE_FILL, ec="k", lw=0.5, zorder=2)

    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls in ((r_in, "-"), (r_mean, (0, (5, 4))), (r_out, "-")):
        ax.plot(r * np.cos(th), r * np.sin(th), color="0.25", ls=ls,
                lw=0.9, zorder=1)

    # diameter callouts down the -y axis at three angles to avoid collision
    for r, lab, ang in ((r_in, r"$d_i$", -90), (r_mean, r"$d_m$", -65),
                        (r_out, r"$d_o$", -50)):
        a = np.deg2rad(ang)
        ax.annotate(lab, xy=(r * np.cos(a), r * np.sin(a)),
                    xytext=(1.32 * r_out * np.cos(a), 1.32 * r_out * np.sin(a)),
                    fontsize=11, ha="center", va="center", color="0.2",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    ax.text(0, 1.16 * r_out, f"{N_BLADES} blades", fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
def main():
    t = tf.turbine_design()
    moc = tf.moc_design()
    bx, by, Ru = blade_polygon(moc)

    print(f"  design pt: u={t.u:.1f}  c3={t.c3:.1f}  c3u={t.c3u:.1f}  "
          f"c3m={t.c3m:.1f}  c4u={t.c4u:.1f}  c4={t.c4:.1f}  "
          f"beta={np.rad2deg(t.beta):.1f} deg")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                             gridspec_kw=dict(width_ratios=[0.9, 2.0, 1.5]))
    panel_le(axes[0], bx, by, Ru)
    panel_triangles(axes[1], t)
    panel_ring(axes[2], bx, by)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
    fig.savefig("proto_cantilever_fig.png", dpi=160, bbox_inches="tight",
                pad_inches=0.03)
    print("  wrote proto_cantilever_fig.png")


if __name__ == "__main__":
    main()
