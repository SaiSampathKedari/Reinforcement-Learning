"""Interactive Plotly figures for Windy Gridworld (S&B Example 6.5).

Clean editorial style: each figure carries a bold title, a grey "how to read
this" subtitle, and rich hover tooltips so it explains itself. Hover any cell
or point to read exact values; the policy-evolution figure animates with a
slider + play button.

Figures:
    plot_gridworld          -- the task layout (start, goal, per-column wind)
    plot_learning_curve     -- episodes completed vs time steps (the Example 6.5 plot)
    plot_steps_per_episode  -- episode length over training (log scale)
    plot_greedy_policy      -- greedy action per cell, as rotated arrows
    plot_state_values       -- V(s) = max_a Q(s, a) heatmap
    plot_greedy_trajectory  -- the greedy start-to-goal path; returns (fig, n_steps)
    plot_value_of_start     -- V(start) climbing over training
    plot_policy_churn       -- how much the greedy policy changes per episode
    plot_policy_evolution   -- ANIMATED greedy policy forming over training

Coordinate convention matches the env: cell (row, col), row 0 at the TOP. The
y-axis is reversed so row 0 is drawn on top and an "up" action points up.
"""

from __future__ import annotations
import math
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

# --- editorial palette ---
FONT = "Inter, Helvetica Neue, Arial, sans-serif"
INK = "#1F2933"        # primary text
SUBTLE = "#6B7280"     # subtitle / secondary
GRID = "#E5E7EB"       # gridlines
ACCENT = "#4C6FE7"     # primary data (indigo)
ACCENT2 = "#E45756"    # reference / optimal (coral)
START_C = "#0F9D8F"    # start marker (teal)
GOAL_C = "#E45756"     # goal marker (coral)
ARROW_C = "#334155"    # policy arrows (slate)
WIND_SCALE = [[0.0, "#FFFFFF"], [0.5, "#E3EEF8"], [1.0, "#A7C7E7"]]
VALUE_SCALE = "Viridis"

# action (drow, dcol) -> short label, for hover
_NAME_BY_DELTA = {
    (-1, 0): "up", (1, 0): "down", (0, 1): "right", (0, -1): "left",
    (-1, 1): "up-right", (1, 1): "down-right", (1, -1): "down-left", (-1, -1): "up-left",
    (0, 0): "stay",
}


# ---------------------------------------------------------------------------
# Styling + grid helpers
# ---------------------------------------------------------------------------

def _editorial(fig, title, subtitle, height=560, width=None):
    """Apply the clean editorial layout: titled, subtitled, light, spacious."""
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:{SUBTLE}'>{subtitle}</span>",
            x=0.04, xanchor="left", y=0.94, yanchor="top",
            font=dict(size=21, color=INK, family=FONT),
        ),
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=64, r=44, t=104, b=66),
        height=height,
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="white"),
    )
    if width is not None:
        fig.update_layout(width=width)
    return fig


def _grid_axes(fig, env):
    """Square cells, row 0 on top, no chartjunk.

    `constrain="domain"` makes the locked-aspect axes shrink the *drawing box* to
    the data instead of padding the *range* -- so columns stay 0..9 (not -5..14)
    and the grid sits tight rather than stretched across a wide canvas.
    """
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="", title_text="column",
                     dtick=1, tickfont=dict(color=SUBTLE),
                     range=[-0.5, env.N_COLS - 0.5], constrain="domain")
    fig.update_yaxes(showgrid=False, zeroline=False, ticks="", title_text="row (0 = top)",
                     dtick=1, tickfont=dict(color=SUBTLE),
                     range=[env.N_ROWS - 0.5, -0.5], constrain="domain",
                     scaleanchor="x", scaleratio=1)


def _wind_heatmap(env):
    """Continuous (gap-less) background shading each column by its wind strength.

    Gap-less so the cell borders come from `_grid_lines` instead -- otherwise
    calm (white) columns would have no visible boundary.
    """
    z = [[float(env.WIND[c]) for c in range(env.N_COLS)] for _ in range(env.N_ROWS)]
    return go.Heatmap(
        z=z, x=list(range(env.N_COLS)), y=list(range(env.N_ROWS)),
        colorscale=WIND_SCALE, zmin=0, zmax=max(env.WIND), xgap=0, ygap=0,
        showscale=False,
        hovertemplate="cell (row %{y}, col %{x})<br>wind: %{z:.0f} ↑<extra></extra>",
    )


def _grid_lines(env):
    """Full cell-boundary lattice over the whole grid, so EVERY cell is visible.

    Drawn as one line trace (segments separated by None). Add it right after the
    wind fill and before arrows/markers so the lattice sits above the shading but
    below the foreground.
    """
    xs, ys = [], []
    for c in range(env.N_COLS + 1):                 # vertical lines
        xs += [c - 0.5, c - 0.5, None]
        ys += [-0.5, env.N_ROWS - 0.5, None]
    for r in range(env.N_ROWS + 1):                 # horizontal lines
        xs += [-0.5, env.N_COLS - 0.5, None]
        ys += [r - 0.5, r - 0.5, None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#64748B", width=1),
                      hoverinfo="skip", showlegend=False)


def _sg_markers(env):
    """Start (S) and Goal (G) markers with labels."""
    sr, sc = env.START
    gr, gc = env.GOAL
    s = go.Scatter(x=[sc], y=[sr], mode="markers+text", text=["S"],
                   textposition="middle center", textfont=dict(color="white", size=14, family=FONT),
                   marker=dict(size=30, color=START_C, line=dict(color="white", width=1.5)),
                   hovertemplate="Start (row %{y}, col %{x})<extra></extra>", showlegend=False)
    g = go.Scatter(x=[gc], y=[gr], mode="markers+text", text=["G"],
                   textposition="middle center", textfont=dict(color="white", size=14, family=FONT),
                   marker=dict(size=30, color=GOAL_C, line=dict(color="white", width=1.5)),
                   hovertemplate="Goal (row %{y}, col %{x})<extra></extra>", showlegend=False)
    return [s, g]


def _add_wind_numbers(fig, env):
    """Wind strength printed under each column (paper-anchored, like the book)."""
    fig.add_annotation(x=-0.5, y=-0.085, xref="x", yref="paper", text="wind ↑",
                       showarrow=False, font=dict(color=SUBTLE, size=11), xanchor="right")
    for c, w in enumerate(env.WIND):
        fig.add_annotation(x=c, y=-0.085, xref="x", yref="paper", text=f"<b>{w}</b>",
                           showarrow=False, font=dict(color=(SUBTLE if w == 0 else ACCENT), size=12))


def _angle(drow, dcol):
    """Heading in degrees clockwise from north (up), for marker rotation."""
    return math.degrees(math.atan2(dcol, -drow))


def _arrow_scatter(env, pi, name="greedy action", showlegend=False):
    """One Scatter of rotated arrow markers, one per non-terminal, non-stay cell."""
    goal = env._encode(*env.GOAL)
    xs, ys, angs, names = [], [], [], []
    for s in range(env.n_states):
        if s == goal:
            continue
        r, c = env._decode(s)
        drow, dcol = env._deltas[int(pi[s])]
        if (drow, dcol) == (0, 0):
            continue
        xs.append(c); ys.append(r)
        angs.append(_angle(drow, dcol))
        names.append(_NAME_BY_DELTA.get((drow, dcol), "?"))
    return go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(symbol="arrow", size=15, angle=angs, angleref="up",
                    color=ARROW_C, line=dict(width=0)),
        text=names, hovertemplate="(row %{y}, col %{x}) → <b>%{text}</b><extra></extra>",
        name=name, showlegend=showlegend,
    )


def _value_labels(env, V):
    """Per-cell V(s) numbers as a text trace, colored for contrast on the colorscale.

    Black text on bright cells, white on dark -- so every number stays readable.
    """
    vmin, vmax = float(V.min()), float(V.max())
    span = (vmax - vmin) or 1.0
    xs, ys, texts, norms = [], [], [], []
    for r in range(env.N_ROWS):
        for c in range(env.N_COLS):
            xs.append(c)
            ys.append(r)
            texts.append(f"{V[r, c]:.0f}")
            norms.append((V[r, c] - vmin) / span)
    rgbs = pcolors.sample_colorscale(VALUE_SCALE, norms)

    def _lum(rgb):
        r, g, b = (int(x) for x in rgb[rgb.find("(") + 1:-1].split(","))
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    colors = ["black" if _lum(s) > 0.6 else "white" for s in rgbs]
    return go.Scatter(x=xs, y=ys, mode="text", text=texts,
                      textfont=dict(size=12, color=colors),
                      hoverinfo="skip", showlegend=False)


def greedy_rollout(env, pi, max_steps=200):
    """Trace the greedy policy from START to the goal under deterministic wind.

    Returns the list of visited cells [(row, col), ...].
    """
    rng = np.random.default_rng(0)  # wind deterministic -> rng unused
    s = env._encode(*env.START)
    cells = [env._decode(s)]
    for _ in range(max_steps):
        if s == env._encode(*env.GOAL):
            break
        s, _, terminated = env.step(s, int(pi[s]), rng)
        cells.append(env._decode(s))
        if terminated:
            break
    return cells


def _path_scatter(env, pi, max_steps=120, showlegend=False):
    """The greedy start->goal path as a red line, for overlaying on the grid.

    `max_steps` caps early self-looping policies so the line doesn't tangle.
    """
    cells = greedy_rollout(env, pi, max_steps=max_steps)
    reached = cells[-1] == env._decode(env._encode(*env.GOAL))
    return go.Scatter(
        x=[c for _, c in cells], y=[r for r, _ in cells],
        mode="lines", line=dict(color=ACCENT2, width=3),
        opacity=0.95 if reached else 0.55,
        hoverinfo="skip", showlegend=showlegend, name="greedy path (red)")


def _eps_greedy_rollout(env, Q, epsilon=0.1, max_steps=250, seed=1):
    """One epsilon-greedy rollout from START using Q. Stochastic but seeded.

    Represents the behavior policy that actually runs episodes: 90% greedy,
    10% random. From an immature Q this wanders a lot before reaching G (or
    hits the cap), mirroring the long early episodes during training.
    """
    rng = np.random.default_rng(seed)
    goal = env._encode(*env.GOAL)
    s = env._encode(*env.START)
    cells = [env._decode(s)]
    for _ in range(max_steps):
        if s == goal:
            break
        a = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(Q[s].argmax())
        s, _, terminated = env.step(s, a, rng)
        cells.append(env._decode(s))
        if terminated:
            break
    return cells


def _eps_path_scatter(env, Q, showlegend=False):
    """epsilon-greedy rollout as a yellow line (the exploring behavior policy)."""
    cells = _eps_greedy_rollout(env, Q)
    return go.Scatter(
        x=[c for _, c in cells], y=[r for r, _ in cells],
        mode="lines", line=dict(color="#EAB308", width=2),
        opacity=0.6, hoverinfo="skip", showlegend=showlegend,
        name="ε-greedy path (yellow)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_gridworld(env):
    """The task: where you start, where you're going, and where the wind blows."""
    fig = go.Figure([_wind_heatmap(env), _grid_lines(env), *_sg_markers(env)])
    _grid_axes(fig, env)
    _add_wind_numbers(fig, env)
    _editorial(
        fig, "Windy Gridworld — the task",
        "Move S → G in as few steps as possible (reward −1 per step). Shaded columns "
        "push you upward by the number shown. Hover any cell for its wind.",
        height=540, width=640,
    )
    return fig


def plot_learning_curve(steps_per_episode, optimal_steps=15):
    """Episodes completed vs time steps — the defining Example 6.5 plot."""
    steps = np.asarray(steps_per_episode)
    cum = np.cumsum(steps)
    episodes = np.arange(1, len(steps) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, cum[-1]], y=[0, cum[-1] / optimal_steps], mode="lines",
        line=dict(color=SUBTLE, width=1.5, dash="dash"),
        name=f"fastest possible (1 / {optimal_steps})",
        hovertemplate="optimal pace<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=cum, y=episodes, mode="lines", line=dict(color=ACCENT, width=2.5),
        name="Sarsa",
        hovertemplate="time step %{x:,}<br>episode %{y}<extra></extra>"))
    fig.update_xaxes(gridcolor=GRID, title_text="time steps", range=[0, cum[-1]])
    fig.update_yaxes(gridcolor=GRID, title_text="episodes completed", range=[0, len(steps)])
    fig.update_layout(legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)", bordercolor=GRID, borderwidth=1))
    _editorial(
        fig, "Sarsa reaches the goal faster over time",
        "Each step right = one time step; each step up = one finished episode. A "
        "<b>steeper</b> slope means shorter episodes. The dashed line is the 15-step optimum.",
    )
    return fig


def plot_steps_per_episode(steps_per_episode, optimal_steps=15, window=10):
    """Episode length over training (log scale), with a smoothed trend."""
    steps = np.asarray(steps_per_episode, dtype=float)
    episodes = np.arange(1, len(steps) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=steps, mode="markers",
        marker=dict(color=ACCENT, size=4, opacity=0.35), name="per episode",
        hovertemplate="episode %{x}<br>%{y:.0f} steps<extra></extra>"))
    if len(steps) >= window:
        ma = np.convolve(steps, np.ones(window) / window, mode="valid")
        fig.add_trace(go.Scatter(
            x=episodes[window - 1:], y=ma, mode="lines",
            line=dict(color=ACCENT, width=2.5), name=f"{window}-episode average",
            hovertemplate="episode %{x}<br>avg %{y:.1f} steps<extra></extra>"))
    fig.add_hline(y=optimal_steps, line=dict(color=ACCENT2, width=1.5, dash="dash"),
                  annotation_text=f"optimal = {optimal_steps}", annotation_position="bottom right",
                  annotation_font_color=ACCENT2)
    # Explicit plain-integer log ticks (avoids Plotly's SI auto-labels like "100T").
    mx = float(steps.max())
    ticks = [t for t in (15, 30, 50, 100, 300, 1000, 3000, 10000) if t <= mx * 1.25]
    fig.update_xaxes(gridcolor=GRID, title_text="episode")
    fig.update_yaxes(gridcolor=GRID, title_text="steps to reach goal (log scale)", type="log",
                     tickmode="array", tickvals=ticks, ticktext=[str(t) for t in ticks],
                     range=[np.log10(max(10, optimal_steps - 3)), np.log10(mx * 1.25)])
    fig.update_layout(legend=dict(x=0.98, y=0.98, xanchor="right", bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor=GRID, borderwidth=1))
    _editorial(
        fig, "Episodes get shorter as Sarsa learns",
        "Each dot is one episode's length (log scale). The line is a moving average; it "
        "falls toward the 15-step optimum and then hovers a little above it.",
    )
    return fig


def plot_greedy_policy(env, pi):
    """The greedy action in every cell, drawn as a rotated arrow."""
    fig = go.Figure([_wind_heatmap(env), _grid_lines(env), _arrow_scatter(env, pi), *_sg_markers(env)])
    _grid_axes(fig, env)
    _add_wind_numbers(fig, env)
    _editorial(
        fig, "The learned greedy policy",
        "Each arrow is the best action (argmax Q) in that cell. Watch it lean down/right "
        "through the windy columns so the upward wind nets out to forward progress.",
        height=540, width=640,
    )
    return fig


def plot_state_values(env, Q):
    """V(s) = max_a Q(s, a), shown two ways: top-down heatmap + rotatable 3D surface."""
    V = Q.max(axis=1).reshape(env.N_ROWS, env.N_COLS)
    cols, rows = list(range(env.N_COLS)), list(range(env.N_ROWS))
    pi = Q.argmax(axis=1)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.46, 0.54],
        specs=[[{"type": "xy"}, {"type": "surface"}]],
        subplot_titles=("top-down heatmap", "3-D surface — drag to rotate"),
        horizontal_spacing=0.04,
    )

    # Left: heatmap + grid lattice + S/G markers.
    fig.add_trace(go.Heatmap(
        z=V, x=cols, y=rows, colorscale=VALUE_SCALE, xgap=0, ygap=0, showscale=False,
        hovertemplate="cell (row %{y}, col %{x})<br>V = %{z:.2f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(_value_labels(env, V), row=1, col=1)   # readable per-cell numbers
    fig.add_trace(_grid_lines(env), row=1, col=1)
    fig.add_trace(_path_scatter(env, pi), row=1, col=1)   # greedy path (red)
    for t in _sg_markers(env):
        fig.add_trace(t, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="", title_text="column",
                     dtick=1, tickfont=dict(color=SUBTLE), range=[-0.5, env.N_COLS - 0.5],
                     constrain="domain", row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, ticks="", title_text="row (0 = top)",
                     dtick=1, tickfont=dict(color=SUBTLE), range=[env.N_ROWS - 0.5, -0.5],
                     constrain="domain", scaleanchor="x", scaleratio=1, row=1, col=1)

    # Right: 3D surface (height = value), with a contour projection on the floor.
    fig.add_trace(go.Surface(
        z=V, x=cols, y=rows, colorscale=VALUE_SCALE,
        colorbar=dict(title="V(s)", thickness=14, outlinewidth=0, x=1.0),
        hovertemplate="cell (row %{y}, col %{x})<br>V = %{z:.2f}<extra></extra>",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True))),
        row=1, col=2)

    # The greedy path, walked along the surface (lifted slightly so it sits on top).
    path = greedy_rollout(env, pi)
    lift = 0.03 * (V.max() - V.min() or 1.0)
    fig.add_trace(go.Scatter3d(
        x=[c for _, c in path], y=[r for r, _ in path],
        z=[V[r, c] + lift for r, c in path],
        mode="lines+markers", line=dict(color=ACCENT2, width=6),
        marker=dict(size=3, color=ACCENT2),
        customdata=list(range(len(path))),
        hovertemplate="greedy step %{customdata}<br>(row %{y}, col %{x})<extra></extra>",
        name="greedy path", showlegend=False),
        row=1, col=2)

    fig.update_scenes(
        xaxis_title="column", yaxis_title="row", zaxis_title="V(s)",
        yaxis=dict(autorange="reversed"),
        camera=dict(eye=dict(x=1.6, y=1.5, z=1.1)),
        aspectmode="manual", aspectratio=dict(x=1.4, y=1.0, z=0.7),
    )

    _editorial(
        fig, "How good is each cell?  V(s) = max₍ₐ₎ Q(s, a)",
        "Two views of the same value function: brighter / higher = closer to the goal "
        "(≈ −steps remaining). The ridges and bumps are the wind warping distance-to-goal; "
        "the red trail is the greedy path climbing the hill toward G. Drag to rotate.",
        height=560, width=1040,
    )
    return fig


def plot_greedy_trajectory(env, pi):
    """The greedy path from S to G. Returns (fig, n_steps)."""
    cells = greedy_rollout(env, pi)
    n_steps = len(cells) - 1
    ys = [r for r, _ in cells]
    xs = [c for _, c in cells]

    fig = go.Figure([_wind_heatmap(env), _grid_lines(env)])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=ACCENT2, width=3), marker=dict(size=8, color=ACCENT2),
        customdata=list(range(len(cells))),
        hovertemplate="step %{customdata}<br>cell (row %{y}, col %{x})<extra></extra>",
        showlegend=False))
    for t in _sg_markers(env):
        fig.add_trace(t)
    _grid_axes(fig, env)
    _add_wind_numbers(fig, env)
    _editorial(
        fig, f"Greedy trajectory — {n_steps} steps",
        "The path you get by always taking the greedy action from S. With the optimal "
        "policy this is the 15-step minimum. Hover a point for its step number.",
        height=540, width=640,
    )
    return fig, n_steps


def plot_value_of_start(snapshots, env, optimal_return=-15):
    """V(start) over training — how the agent's estimate of the start improves."""
    start = env._encode(*env.START)
    episodes = [k for k, _ in snapshots]
    v_start = [Q[start].max() for _, Q in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=v_start, mode="lines", line=dict(color=ACCENT, width=2.5),
        name="V(start)", hovertemplate="episode %{x}<br>V(start) = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=optimal_return, line=dict(color=ACCENT2, width=1.5, dash="dash"),
                  annotation_text=f"greedy optimum = {optimal_return}", annotation_position="top right",
                  annotation_font_color=ACCENT2)
    fig.update_xaxes(gridcolor=GRID, title_text="episode")
    fig.update_yaxes(gridcolor=GRID, title_text="V(start) = max₍ₐ₎ Q(S, a)")
    _editorial(
        fig, "The agent learns how far the start is from the goal",
        "V(start) ≈ −(steps to reach G). It starts optimistic at 0, sinks as the −1/step "
        "cost is learned, and settles just below the −15 greedy optimum — the gap is the "
        "ε-greedy exploration tax (same gap as steps-per-episode settling above 15).",
    )
    return fig


def plot_policy_churn(snapshots):
    """How many cells change their greedy action between consecutive snapshots."""
    episodes = [k for k, _ in snapshots]
    pis = [Q.argmax(axis=1) for _, Q in snapshots]
    churn = [int((pis[i] != pis[i - 1]).sum()) for i in range(1, len(pis))]

    fig = go.Figure(go.Scatter(
        x=episodes[1:], y=churn, mode="lines", line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(76,111,231,0.12)",
        hovertemplate="episode %{x}<br>%{y} cells changed action<extra></extra>"))
    fig.update_xaxes(gridcolor=GRID, title_text="episode")
    fig.update_yaxes(gridcolor=GRID, title_text="cells whose best action changed", rangemode="tozero")
    _editorial(
        fig, "The policy stops changing — that's convergence",
        "How many cells flipped their greedy action since the previous episode. Large "
        "early (the policy is being rewritten), decaying toward 0 as Q settles.",
    )
    return fig


def plot_policy_evolution(env, snapshots, max_frames=40):
    """ANIMATED: the greedy policy forming over training (slider + play button)."""
    idx = np.unique(np.linspace(0, len(snapshots) - 1, min(max_frames, len(snapshots))).astype(int))
    chosen = [snapshots[i] for i in idx]

    base_Q = chosen[0][1]
    base_pi = base_Q.argmax(axis=1)
    fig = go.Figure([
        _wind_heatmap(env), _grid_lines(env),
        _eps_path_scatter(env, base_Q, showlegend=True),       # 2: yellow ε-greedy path
        _path_scatter(env, base_pi, showlegend=True),          # 3: red greedy path
        *_sg_markers(env),                                     # 4, 5
        _arrow_scatter(env, base_pi, showlegend=True),         # 6: arrows
    ])
    eps_idx, greedy_idx, arrow_idx = 2, 3, len(fig.data) - 1
    _grid_axes(fig, env)
    _add_wind_numbers(fig, env)

    # Each frame updates all three: ε-greedy path, greedy path, and arrow field.
    frames = [
        go.Frame(name=str(k),
                 data=[_eps_path_scatter(env, Q), _path_scatter(env, Q.argmax(axis=1)),
                       _arrow_scatter(env, Q.argmax(axis=1))],
                 traces=[eps_idx, greedy_idx, arrow_idx])
        for k, Q in chosen
    ]
    fig.frames = frames

    steps = [dict(method="animate", label=str(k),
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k, _ in chosen]
    fig.update_layout(
        sliders=[dict(active=0, x=0.04, len=0.92, pad=dict(t=58, b=8),
                      currentvalue=dict(prefix="Episode: ", font=dict(size=14, color=INK)),
                      steps=steps)],
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.04, y=1.13, xanchor="left",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=350, redraw=True),
                                                    fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=True))]),
                          ])],
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=11)),
    )
    _editorial(
        fig, "Watch the policy form: ε-greedy vs greedy",
        "Drag the slider / press play. <b>Yellow</b> = ε-greedy rollout (the exploring "
        "behavior that actually runs episodes — long & wandering early, always reaches G). "
        "<b>Red</b> = greedy path (deterministic argmax Q — loops early, then snaps to the "
        "15-step optimum). Arrows = best action per cell.",
        height=620, width=680,
    )
    return fig
