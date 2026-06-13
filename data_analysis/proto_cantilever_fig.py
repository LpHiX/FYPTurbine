"""
proto_cantilever_fig.py — PROTOTYPE (not wired into the thesis build).

Three-panel borderless terminology figure for the radial-inflow cantilever
turbine. MoC frame convention (verified against turbinemoc.generate):
    local x = meridional (chord / through-flow),  local y = tangential (u / pitch).
The blade is a symmetric impulse bucket; LE = one foot, TE = the other.

  (L) velocity triangles overlaid on the MoC blade contour. u (tangential)
      horizontal, meridional vertical, so the contour reads lower-edge-left /
      upper-edge-right. Inlet (3) triangle hung on the LE, exit (4) on the TE.
      Real design-point numbers. beta angles are NOT on the triangles.
  (M) the same contour with the LE and TE straight segments extended to draw
      beta_3 / beta_4 against the horizontal (u).
  (R) annular cascade: d_i 85 / d_m 95 / d_o 105 mm, 50 MoC blades staggered
      87 deg clockwise.

Run:  python proto_cantilever_fig.py   ->  proto_cantilever_fig.png
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thesis_figures as tf

# ----------------------------------------------------------------------------- #
# knobs
# ----------------------------------------------------------------------------- #
N_BLADES   = 50
D_IN_MM, D_MEAN_MM, D_OUT_MM = 85.0, 95.0, 105.0
STAGGER_CW = 87.0                          # blade rotation in the ring [deg, CW]
BLADE_FILL = "0.85"


# ----------------------------------------------------------------------------- #
# single solid MoC blade silhouette (the _draw_passage crescent).
#   native frame: x = meridional, y = tangential. LE/TE are the two feet.
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
    return bx, by


def rot(px, py, deg):
    """rotate points by +deg (CCW)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return px * c - py * s, px * s + py * c


def oriented_blade(bx, by, chord_units):
    """blade for the L/M panels: u (local y) -> horizontal, meridional (local
    x) -> vertical (LE up, TE down), lower-edge-left / upper-edge-right.
    Mapping (x,y)->(y,-x), centred, scaled so the LE-TE span = chord_units."""
    X, Y = by.copy(), -bx.copy()
    X -= X.mean()
    Y -= Y.mean()
    s = chord_units / (Y.max() - Y.min())
    return X * s, Y * s


# ----------------------------------------------------------------------------- #
# panel L — velocity triangles on the blade
# ----------------------------------------------------------------------------- #
def _arrow(ax, p0, p1, color, lw=1.6):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def _triangle_at(ax, A, U, W, suf, cu_lab, wu_lab):
    """One velocity triangle anchored at the u-tip A (the w_u/w corner).
    u : O->A (O = A - U),  w : A->C (C = A + W),  c : O->C. Component guides."""
    cU, cC, cW, cK = "0.2", "C0", "C3", "0.55"
    O = A - U
    C = A + W
    K = np.array([C[0], O[1]])                    # right-angle corner (w_u end)
    right = C[0] >= A[0]
    _arrow(ax, O, A, cU)                          # u
    _arrow(ax, A, C, cW)                          # w
    _arrow(ax, O, C, cC)                          # c
    ax.text((O[0] + A[0]) / 2, O[1] + 6, r"$u$", color=cU, fontsize=10,
            ha="center", va="bottom")
    ax.text((A[0] + C[0]) / 2 + (5 if right else -5), (A[1] + C[1]) / 2,
            r"$w_%s$" % suf, color=cW, fontsize=11,
            ha="left" if right else "right", va="center")
    ax.text((O[0] + C[0]) / 2 + (5 if right else -5), (O[1] + C[1]) / 2 - 6,
            r"$c_%s$" % suf, color=cC, fontsize=11,
            ha="left" if right else "right", va="center")
    ax.plot([O[0], K[0]], [O[1], K[1]], color=cK, lw=0.7, ls=":")
    ax.plot([K[0], K[0]], [O[1], C[1]], color=cK, lw=0.7, ls=":")
    ax.text((O[0] + K[0]) / 2, O[1] + 6, cu_lab, color=cK, fontsize=9,
            ha="center", va="bottom")
    ax.text(K[0] + (4 if right else -4), (O[1] + C[1]) / 2, r"$c_{%s m}$" % suf,
            color=cK, fontsize=9, ha="left" if right else "right", va="center")
    ax.text((A[0] + K[0]) / 2, A[1] - 6, wu_lab, color=cK, fontsize=9,
            ha="center", va="top")


def _beta_at(ax, A, W, lab, r=26):
    """arc the angle between W and the horizontal at the anchor A (= the metal
    angle, since w lies along the blade LE/TE tangent)."""
    base = 0.0 if W[0] >= 0 else np.pi
    ang = np.arctan2(W[1], W[0])
    d = (ang - base + np.pi) % (2 * np.pi) - np.pi
    th = np.linspace(base, base + d, 30)
    ax.plot([A[0], A[0] + 1.25 * r * np.cos(base)], [A[1], A[1]],
            color="0.55", lw=0.8, ls="--")
    ax.plot(A[0] + r * np.cos(th), A[1] + r * np.sin(th), color="C3", lw=1.1)
    am = base + d / 2
    ax.text(A[0] + 1.7 * r * np.cos(am), A[1] + 1.7 * r * np.sin(am), lab,
            color="C3", fontsize=12, ha="center", va="center")


def panel_blade_triangles(ax, t, X, Y):
    """Combined plot: the MoC blade contour with the velocity triangles overlaid
    and aligned to it. Inlet (3) anchored (u-tip) on the LE, exit (4) on the TE;
    w3 runs along the LE tangent, w4 along the TE tangent, and beta_3 / beta_4
    are the angles those make with the horizontal (u). True scale."""
    vs = 0.20                                    # m/s -> figure units (true scale)
    u, c3u, c3m = float(t.u), float(t.c3u), float(t.c3m)
    c4u = float(t.c4u)
    ax.fill(X, Y, color=BLADE_FILL, ec="k", lw=1.3, zorder=1)
    le = np.array([X[np.argmax(Y)], Y.max()])    # LE foot (top)
    te = np.array([X[np.argmin(Y)], Y.min()])    # TE foot (bottom)
    U = np.array([u, 0.0]) * vs
    W3 = np.array([c3u - u, -c3m]) * vs           # along LE tangent
    W4 = np.array([c4u - u, -c3m]) * vs           # along TE tangent

    # displace each triangle horizontally to the LEFT, clear of the blade;
    # dashed leader connects the displaced u-tip back to the foot.
    dx = 70.0
    A3 = le - np.array([dx, 0.0])
    A4 = te - np.array([dx, 0.0])
    ax.plot([le[0], A3[0]], [le[1], A3[1]], color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot([te[0], A4[0]], [te[1], A4[1]], color="0.4", ls="--", lw=0.8, zorder=0)

    _triangle_at(ax, A3, U, W3, "3", r"$c_{3u}$", r"$w_{3u}$")
    _triangle_at(ax, A4, U, W4, "4", r"$-c_{4u}$", r"$-w_{4u}$")
    # beta on the blade feet: solid extended LE/TE edge line + arc, to the LEFT
    n = len(X)
    i_le, i_te = int(np.argmax(Y)), int(np.argmin(Y))

    def tang(i):
        a, b = X[(i - 3) % n], X[(i + 3) % n]
        c, d = Y[(i - 3) % n], Y[(i + 3) % n]
        return np.rad2deg(np.arctan2(d - c, b - a))

    _beta(ax, le, tang(i_le), r"$\beta_3$")
    _beta(ax, te, tang(i_te), r"$\beta_4$")
    ax.text(A3[0], A3[1] + 22, "rotor inlet (3)", fontsize=8, ha="center")
    ax.text(A4[0], A4[1] - 26, "rotor exit (4)", fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
# panel M — blade metal angles beta_3 / beta_4
# ----------------------------------------------------------------------------- #
def _beta(ax, foot, tangent_deg, lab, ext=62, r=30):
    """extend the LE/TE straight segment to the LEFT of `foot` only, draw a
    leftward horizontal reference, and arc the angle between them."""
    d = np.deg2rad(tangent_deg)
    dx, dy = np.cos(d), np.sin(d)
    if dx > 0:                                  # force the line to point LEFT
        dx, dy = -dx, -dy
    p1 = foot + ext * np.array([dx, dy])
    ax.plot([foot[0], p1[0]], [foot[1], p1[1]], color="0.2", lw=1.2)
    ax.plot([foot[0], foot[0] - ext * 0.95], [foot[1], foot[1]],
            color="0.55", lw=0.8, ls="--")
    # small signed deviation of the leftward line from the leftward horizontal
    da = np.arctan2(dy, dx) - np.pi
    da = (da + np.pi) % (2 * np.pi) - np.pi      # wrap to (-pi, pi]
    th = np.linspace(np.pi, np.pi + da, 40)
    ax.plot(foot[0] + r * np.cos(th), foot[1] + r * np.sin(th),
            color="C3", lw=1.1)
    am = np.pi + da / 2
    ax.text(foot[0] + 1.7 * r * np.cos(am), foot[1] + 1.7 * r * np.sin(am),
            lab, color="C3", fontsize=12, ha="center", va="center")


def panel_angles(ax, moc, X, Y):
    ax.fill(X, Y, color=BLADE_FILL, ec="k", lw=1.3, zorder=1)
    le = np.array([X[np.argmax(Y)], Y.max()])
    te = np.array([X[np.argmin(Y)], Y.min()])
    # tangent of the contour at each foot, in the oriented frame
    i_le = int(np.argmax(Y))
    i_te = int(np.argmin(Y))
    n = len(X)
    def tang(i):
        a, b = X[(i - 3) % n], X[(i + 3) % n]
        c, d = Y[(i - 3) % n], Y[(i + 3) % n]
        return np.rad2deg(np.arctan2(d - c, b - a))
    _beta(ax, le, tang(i_le), r"$\beta_3$")
    _beta(ax, te, tang(i_te), r"$\beta_4$")
    ax.text((le[0] + te[0]) / 2, te[1] - 30, "blade metal angles",
            fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
# panel R — annular cascade
# ----------------------------------------------------------------------------- #
def panel_ring(ax, Xo, Yo):
    """Xo,Yo = oriented blade (LE at +Y, TE at -Y, meridional vertical), the
    same silhouette the middle panel shows. Each blade is placed with its OWN
    radial axis: LE (+Y) -> outer diameter, TE (-Y) -> inner diameter. Uniform
    scale (shape preserved); the only per-blade rotation is the radial heading."""
    r_in, r_mean, r_out = D_IN_MM / 2, D_MEAN_MM / 2, D_OUT_MM / 2
    band = r_out - r_in
    s = 0.98 * band / (Yo.max() - Yo.min())       # meridional span -> radial band
    bx_r, by_r = Xo * s, Yo * s                    # by_r -> radial, bx_r -> tangential

    for k in range(N_BLADES):
        th = 2 * np.pi * k / N_BLADES
        radial = r_mean + by_r                     # LE outer, TE inner
        tang = bx_r
        gx = radial * np.cos(th) - tang * np.sin(th)
        gy = radial * np.sin(th) + tang * np.cos(th)
        ax.fill(gx, gy, color=BLADE_FILL, ec="k", lw=0.5, zorder=2)

    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls in ((r_in, "-"), (r_mean, (0, (5, 4))), (r_out, "-")):
        ax.plot(r * np.cos(th), r * np.sin(th), color="0.25", ls=ls, lw=0.9,
                zorder=1)
    for r, lab, ang in ((r_in, r"$d_i$", -90), (r_mean, r"$d_m$", -66),
                        (r_out, r"$d_o$", -50)):
        a = np.deg2rad(ang)
        ax.annotate(lab, xy=(r * np.cos(a), r * np.sin(a)),
                    xytext=(1.34 * r_out * np.cos(a), 1.34 * r_out * np.sin(a)),
                    fontsize=11, ha="center", va="center", color="0.2",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    ax.text(0, 1.18 * r_out, f"{N_BLADES} blades", fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
def main():
    t = tf.turbine_design()
    moc = tf.moc_design()
    bx, by = blade_polygon(moc)
    X_tri, Y_tri = oriented_blade(bx, by, chord_units=100)   # left: contour + triangles
    X_ring, Y_ring = oriented_blade(bx, by, chord_units=1.0)  # ring (rescaled inside)

    print(f"  design pt: u={t.u:.1f}  c3={t.c3:.1f}  c3u={t.c3u:.1f}  "
          f"c3m={t.c3m:.1f}  c4u={t.c4u:.1f}  c4={t.c4:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0),
                             gridspec_kw=dict(width_ratios=[1.5, 1.4]))
    panel_blade_triangles(axes[0], t, X_tri, Y_tri)
    panel_ring(axes[1], -X_ring, Y_ring)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
    fig.savefig("proto_cantilever_fig.png", dpi=160, bbox_inches="tight",
                pad_inches=0.03)
    print("  wrote proto_cantilever_fig.png")


if __name__ == "__main__":
    main()
