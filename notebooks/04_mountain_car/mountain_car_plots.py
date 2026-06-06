"""Plots and policy animation for Mountain Car (S&B Figure 10.1)."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D

# throttle = action - 1: 0 -> reverse(-1), 1 -> coast(0), 2 -> forward(+1)
ACTION_COLOR = {0: "#D32F2F", 1: "#9E9E9E", 2: "#2E7D32"}
ACTION_NAME = {0: "reverse  (-1)", 1: "coast   ( 0)", 2: "forward (+1)"}


def cost_to_go_grid(q, P, V):
    """Cost-to-go surface -max_a q_hat((pos, vel), a) over a (P x V) grid.

    Returns Z of shape (len(V), len(P)) aligned with `np.meshgrid(P, V)`.
    """
    n_actions = 3
    return np.array([[-max(q.value((p, v), a) for a in range(n_actions))
                      for p in P] for v in V])


def plot_figure_10_1(P, V, surfaces, ncols=2, cmap="viridis"):
    """Reproduce S&B Fig 10.1: cost-to-go surfaces learned during one run.

    `surfaces` is an ordered dict {label: Z}, each Z of shape (len(V), len(P))
    (e.g. from `cost_to_go_grid`) -- one 3-D colored surface per snapshot. The
    peak sits over the valley-at-rest and slopes to ~0 at the goal. Returns the
    matplotlib figure.
    """
    Pg, Vg = np.meshgrid(P, V)
    n = len(surfaces)
    nrows = -(-n // ncols)                                  # ceil division
    fig = plt.figure(figsize=(5.6 * ncols, 4.7 * nrows))

    for i, (label, Z) in enumerate(surfaces.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_surface(Pg, Vg, Z, cmap=cmap, rstride=1, cstride=1,
                        edgecolor="0.25", linewidth=0.15, antialiased=True)
        # Goal edge (right wall, position = P[-1]) in green, as in the book.
        ax.plot(np.full_like(V, P[-1]), V, np.zeros_like(V),
                color="#2E7D32", lw=4, zorder=10)

        ax.set_title(label, fontsize=12, pad=-2)
        ax.set_xlabel("Position", labelpad=4)
        ax.set_ylabel("Velocity", labelpad=4)
        ax.set_zlabel("cost-to-go", labelpad=2)
        ax.set_xlim(P[0], P[-1])
        ax.set_ylim(V[0], V[-1])
        ax.set_zlim(bottom=0)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=42, azim=-60)                    # book viewpoint (S&B Fig 10.1)
        ax.set_box_aspect((1, 1, 0.75), zoom=1.1)

    fig.suptitle("Figure 10.1 — Mountain Car cost-to-go "
                 r"$-\max_a \hat q(s,a,\mathbf{w})$ (one run)", fontsize=14)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                        wspace=-0.05, hspace=0.14)
    return fig


def _greedy_rollout(env, q, rng, start, max_steps):
    """Run the greedy policy, returning [(position, velocity, action), ...]."""
    s = env.reset(rng) if start is None else np.array([float(start), 0.0])
    traj = [(float(s[0]), float(s[1]), 1)]            # first action is a placeholder
    for _ in range(max_steps):
        a = q.greedy(s, range(env.n_actions), rng)
        s, _r, done = env.step(s, a, rng)
        traj.append((float(s[0]), float(s[1]), int(a)))
        if done:
            break
    return traj


# Visual height of the hill: scaled sin(3x) (Gymnasium's `*0.45 + 0.55`). The
# *0.45 flattens it so the scene is wide rather than a tall narrow column.
_H_SCALE = 0.45


def _hill(x):
    """Displayed terrain height at position x."""
    return _H_SCALE * np.sin(3 * x) + 0.55


def _hill_slope_deg(x):
    """Tangent angle (degrees) of the displayed hill -- for tilting the car."""
    return np.degrees(np.arctan(_H_SCALE * 3 * np.cos(3 * x)))


def _make_car(ax, scale=1.0):
    """Add a little car (body + cabin + two wheels) at the origin; return its parts.

    Parts are drawn in car-local coordinates (wheels on the ground at local y=0);
    `_place_car` rotates and translates them onto the hill each frame.
    """
    s = scale
    wheels = [Circle((-0.05 * s, 0.018 * s), 0.018 * s, color="0.15", zorder=7),
              Circle((0.05 * s, 0.018 * s), 0.018 * s, color="0.15", zorder=7)]
    body = Rectangle((-0.085 * s, 0.018 * s), 0.17 * s, 0.045 * s, zorder=6)
    cabin = Rectangle((-0.035 * s, 0.060 * s), 0.075 * s, 0.032 * s, zorder=6)
    for p in (*wheels, body, cabin):
        ax.add_patch(p)
    return {"body": body, "cabin": cabin, "wheels": wheels}


def _place_car(car, x, y, angle_deg, ax, color):
    """Rotate/translate the car onto (x, y) at the hill slope, tinting the shell."""
    t = Affine2D().rotate_deg(angle_deg).translate(x, y) + ax.transData
    car["body"].set_color(color)
    car["cabin"].set_color(color)
    for p in (car["body"], car["cabin"], *car["wheels"]):
        p.set_transform(t)


def animate_policy(env, q, rng=None, n_rollouts=3, max_steps=500, hold=10,
                   save_path=None, fps=14):
    """Animate the greedy policy of `q` driving the car over the (scaled) sin(3x) hill.

    Plays `n_rollouts` greedy episodes back-to-back, each from a fresh random
    start (`env.reset`), holding `hold` frames at each goal. A little car rides
    the hill, tilted to the local slope and tinted by its throttle action (red
    reverse / grey coast / green forward), with a throttle arrow and a HUD showing
    the rollout, step, velocity and action. The figure aspect matches the scene so
    it fills the frame; `fps` (default 14) sets the playback speed -- lower it to
    slow down. Returns the `FuncAnimation`; if `save_path` is given it is also
    written there (GIF via the pillow writer).
    """
    rng = np.random.default_rng(0) if rng is None else rng

    # Build several rollouts (random starts) and flatten into one frame list,
    # holding the final frame at each goal so the arrival is visible.
    frames = []                                          # (pos, vel, a, rollout, step)
    for r in range(n_rollouts):
        traj = _greedy_rollout(env, q, rng, None, max_steps)
        for si, (pos, vel, a) in enumerate(traj):
            frames.append((pos, vel, a, r, si))
        for _ in range(hold):
            frames.append((*traj[-1], r, len(traj) - 1))

    P = np.linspace(env.POS_MIN, env.POS_MAX, 300)
    H = _hill(P)
    gx = env.GOAL; gy = _hill(gx)
    y_lo, y_hi = H.min() - 0.06, gy + 0.30                # tight vertical bounds
    x_lo, x_hi = env.POS_MIN, env.POS_MAX

    # Match the figure aspect to the (equal-aspect) scene so it fills the frame.
    width = 10.0
    fig, ax = plt.subplots(figsize=(width, width * (y_hi - y_lo) / (x_hi - x_lo)))
    ax.plot(P, H, color="0.3", lw=2)
    ax.fill_between(P, H, y_lo, color="#D9CBB0")          # ground
    ax.plot([gx, gx], [gy, gy + 0.22], color="0.2", lw=1.6)            # flagpole
    ax.plot(gx, gy + 0.22, marker=">", ms=16, color="#2E7D32")        # flag
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal"); ax.axis("off")

    car = _make_car(ax)
    throttle = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                               mutation_scale=20, lw=2.5, zorder=8)
    ax.add_patch(throttle)
    hud = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top",
                  fontsize=12, family="monospace")

    def update(i):
        pos, vel, a, r, si = frames[i]
        y = _hill(pos)
        _place_car(car, pos, y, _hill_slope_deg(pos), ax, ACTION_COLOR[a])
        if a == 1:                                        # coast: hide the arrow
            throttle.set_positions((pos, y + 0.16), (pos, y + 0.16))
        else:
            dx = 0.18 if a == 2 else -0.18
            throttle.set_positions((pos, y + 0.17), (pos + dx, y + 0.17))
            throttle.set_color(ACTION_COLOR[a])
        hud.set_text(f"rollout {r + 1}/{n_rollouts}\nstep {si:3d}\n"
                     f"vel  {vel:+.3f}\n{ACTION_NAME[a]}")
        return ()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    if save_path is not None:
        anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    return anim
