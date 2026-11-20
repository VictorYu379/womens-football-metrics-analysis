import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from mplsoccer import Pitch

from analysis import spearman


LABELS = {"A": "A: 1.59M men's rows", "C": "C: 4.58M men's rows", "D": "D: women only",
          "C_adapter": "C + Beta adapter", "C_continued": "C + continued (match validation)"}
COLORS = {"A": "#777777", "C": "#0072B2", "D": "#D55E00", "C_adapter": "#009E73", "C_continued": "#CC79A7"}


def short_name(model):
    return "C + adapter" if model == "C_adapter" else model


def save_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def probability_plot(metrics, models):
    figure, axes = plt.subplots(1, 2, figsize=(12, max(2.8, .55 * len(models) + 1.5)), layout="constrained")
    for axis, target in zip(axes, ("scores", "concedes")):
        selected = metrics.loc[metrics.target == target].set_index("model").loc[models]
        axis.scatter(selected.log_loss, np.arange(len(models)), c=[COLORS[model] for model in models], s=60)
        axis.set_yticks(np.arange(len(models)), [LABELS[model] for model in models])
        axis.set_ylim(len(models) - .5, -.5)
        axis.set_title(target)
        axis.set_xlabel("Held-out log loss (lower is better; zoomed axis)")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(axis="x", alpha=.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Outcome prediction on the same 231 women's matches")
    return figure


def action_plot(summary, pairs, min_actions=500):
    eligible = summary.loc[summary.actions.ge(min_actions)]
    order = eligible.loc[(eligible.left == pairs[0][0]) & (eligible.right == pairs[0][1])].sort_values("mean_abs_delta").type_name.tolist()
    goal_types = [name for name in order if name in ("goal", "owngoal")]
    other_types = [name for name in order if name not in ("goal", "owngoal")]
    groups = [(label, names) for label, names in (("Goal atoms", goal_types), ("Other atomic actions", other_types)) if names]
    heights = [max(1.1, len(names) * .35) for _, names in groups]
    figure, axes = plt.subplots(len(groups), 2, figsize=(14, sum(heights) + 2), layout="constrained",
                                gridspec_kw={"height_ratios": heights}, squeeze=False)
    colors = [COLORS[left] for left, _ in pairs]
    for row_index, (group_label, names) in enumerate(groups):
        for pair_index, (left, right) in enumerate(pairs):
            values = eligible.loc[(eligible.left == left) & (eligible.right == right)].set_index("type_name").reindex(names)
            positions = np.arange(len(names)) + (pair_index - (len(pairs) - 1) / 2) * .2
            for axis, metric in zip(axes[row_index], ("mean_delta", "mean_abs_delta")):
                axis.scatter(values[metric], positions, s=25, color=colors[pair_index])
        for axis, metric_label in zip(axes[row_index], ("Mean signed difference", "Mean absolute difference")):
            axis.set_yticks(np.arange(len(names)), names)
            axis.axvline(0, color="#888888", linewidth=.7)
            axis.grid(axis="x", alpha=.2)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(f"{group_label}: {metric_label.lower()}")
            axis.set_xlabel("Atomic-VAEP difference per action")
    handles = [Line2D([], [], linestyle="", marker="o", color=color,
                      label=f"{short_name(left)} minus {short_name(right)}") for color, (left, right) in zip(colors, pairs)]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(pairs), frameon=False)
    figure.suptitle(f"Which action types receive different credit? Non-goalkeepers; n >= {min_actions}\nGoal atoms use separate x scales; differences are not errors against ground truth.")
    return figure


def spatial_plot(summary, pairs, min_actions=20):
    event_types = summary.type_name.drop_duplicates().tolist()
    minimum = np.where(summary.type_name.eq("shot"), 10, min_actions)
    visible = summary.loc[summary.actions.ge(minimum) & summary.mean_delta.notna()]
    if visible.empty:
        raise ValueError("No pitch cells meet the display thresholds")
    limit = float(visible.mean_delta.abs().quantile(.98))
    if not np.isfinite(limit) or limit == 0:
        limit = max(float(visible.mean_delta.abs().max()), .001)
    panels = [(left, right, event_type) for left, right in pairs for event_type in event_types]
    figure, axes = plt.subplots(int(np.ceil(len(panels) / 3)), 3, figsize=(18, 5 * int(np.ceil(len(panels) / 3))),
                                layout="constrained", squeeze=False)
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#333333", linewidth=.8)
    for axis, (left, right, event_type) in zip(axes.ravel(), panels):
        pitch.draw(ax=axis)
        selected = summary.loc[(summary.type_name == event_type) & (summary.left == left) & (summary.right == right)]
        means = selected.pivot(index="row", columns="column", values="mean_delta").to_numpy()
        counts = selected.pivot(index="row", columns="column", values="actions").to_numpy()
        threshold = 10 if event_type == "shot" else min_actions
        image = axis.imshow(np.ma.masked_where(counts < threshold, means), extent=(0, 105, 0, 68), origin="lower",
                            cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=.85, zorder=0, interpolation="nearest")
        axis.set_title(f"{event_type}: {short_name(left)} minus {short_name(right)}")
    for axis in axes.ravel()[len(panels):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label="Mean Atomic-VAEP difference", extend="both", pad=.02)
    figure.suptitle("Action-level spatial differences; attack to the right; shared symmetric color scale")
    figure.supxlabel(f"Shots: at least 10 actions/cell; other types: {min_actions}. Blank is not zero.\n"
                     "Colors clip at the 98th percentile of absolute grid means; tables and player scores are not clipped.", fontsize=10)
    return figure


def player_plot(players, pairs, measure="per90", annotate=False):
    models = sorted({model for pair in pairs for model in pair})
    values = players[[f"{model}_{measure}" for model in models]].to_numpy()
    lower, upper = min(0, values.min()), max(0, values.max())
    padding = max((upper - lower) * .04, 1e-8)
    positions = sorted(players.position_bucket.unique())
    palette = dict(zip(positions, plt.colormaps["tab10"].colors))
    figure, axes = plt.subplots(1, len(pairs), figsize=(max(7, 5 * len(pairs)), 5.5), layout="constrained", squeeze=False)
    for axis, (left, right) in zip(axes.ravel(), pairs):
        for position in positions:
            selected = players.loc[players.position_bucket == position]
            axis.scatter(selected[f"{right}_{measure}"], selected[f"{left}_{measure}"],
                         color=palette[position], s=11, alpha=.65, rasterized=True)
        axis.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
        axis.set(xlim=(lower - padding, upper + padding), ylim=(lower - padding, upper + padding),
                 xlabel=f"{LABELS[right]} / {measure}", ylabel=f"{LABELS[left]} / {measure}")
        axis.set_aspect("equal", adjustable="box")
        correlation = spearman(players[f"{left}_{measure}"], players[f"{right}_{measure}"])
        axis.set_title(f"{short_name(left)} vs {short_name(right)} | Spearman {correlation:.3f}")
        axis.grid(alpha=.15)
        if annotate:
            annotate_score_differences(axis, players, f"{left}_{measure}", f"{right}_{measure}")
    handles = [Line2D([], [], marker="o", linestyle="", color=palette[position], label=position) for position in positions]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(f"Same {len(players)} eligible non-goalkeepers; signed Atomic-VAEP {measure}")
    return figure


def annotate_score_differences(axis, players, left_column, right_column, count=8):
    differences = (players[left_column] - players[right_column]).abs()
    selected = players.loc[differences.nlargest(count).index].sort_values(left_column)
    lower, upper = axis.get_ylim()
    positions = np.clip((selected[left_column].to_numpy() - lower) / (upper - lower), .08, .92)
    gap = min(.085, .8 / max(len(positions) - 1, 1))
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    positions[-1] = min(positions[-1], .92)
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    for position, (_, player) in zip(positions, selected.iterrows()):
        axis.annotate(player.player_name, (player[right_column], player[left_column]), xytext=(.98, position),
                      textcoords="axes fraction", ha="right", va="center", fontsize=7,
                      bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1},
                      arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": .5})


def draw_ridges(axis, grouped_values, means, labels, limits, bins=180, smoothing=7):
    edges = np.linspace(*limits, bins + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    kernel = np.ones(smoothing) / smoothing
    color_limit = max(float(np.max(np.abs(means))), 1e-12)
    norm = plt.Normalize(-color_limit, color_limit)
    for position, (values, mean) in enumerate(zip(grouped_values, means)):
        histogram = np.histogram(values, bins=edges)[0]
        density = np.convolve(histogram, kernel, mode="same").astype(float)
        if density.max() > 0:
            density *= .55 / density.max()
        axis.fill_between(centers, position, position + density, color=plt.colormaps["coolwarm"](norm(mean)), alpha=.75)
        axis.plot(centers, position + density, color="#333333", linewidth=.6)
        if limits[0] <= mean <= limits[1]:
            axis.scatter(mean, position + np.interp(mean, centers, density), color="black", s=15, zorder=4)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set(xlim=limits, ylim=(-.25, len(labels) - .2), yticks=np.arange(len(labels)), yticklabels=labels)
    axis.grid(axis="x", alpha=.15)
    axis.spines[["top", "right", "left"]].set_visible(False)


def action_distribution_plot(actions, summary, group_column, pair, title, display_limit=.01):
    if summary.empty:
        raise ValueError("No action groups meet the display threshold")
    groups = summary[group_column].tolist()
    values = [actions.loc[actions[group_column] == group, "vaep_delta"].to_numpy() for group in groups]
    labels = [f"{group} (n={count:,})" for group, count in zip(groups, summary.actions.astype(int))]
    figure, axis = plt.subplots(figsize=(12, max(6, .35 * len(groups) + 2)), layout="constrained")
    draw_ridges(axis, values, summary.mean_delta.to_numpy(), labels, (-display_limit, display_limit))
    axis.set_xlabel(f"Action Atomic-VAEP: {short_name(pair[0])} minus {short_name(pair[1])}")
    axis.set_title(title)
    outside = sum(int(np.sum(np.abs(group_values) > display_limit)) for group_values in values)
    figure.supxlabel(f"Equal peak heights compare shapes, not counts; black dots mark in-range means.\n"
                     f"Outside +/-{display_limit:g}: {outside:,} / {sum(map(len, values)):,} actions; all remain in statistics.", fontsize=9)
    return figure


def position_distribution_plot(players, summary, pair, measure):
    selected = summary.loc[summary.position != "All eligible"].sort_values(["players", "position"], ascending=[True, False])
    positions = selected.position.tolist()
    left, right = pair
    delta = players[f"{left}_{measure}"] - players[f"{right}_{measure}"]
    lower, upper = delta.quantile([.02, .98])
    if np.isclose(lower, upper):
        lower, upper = lower - .001, upper + .001
    padding = .08 * (upper - lower)
    limits = (lower - padding, upper + padding)
    values = [delta.loc[players.position_bucket.eq(position)].to_numpy() for position in positions]
    labels = [f"{position} (n={count})" for position, count in zip(positions, selected.players)]
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), width_ratios=(.9, 1.35), layout="constrained")
    axes[0].barh(np.arange(len(positions)), selected.spearman.clip(lower=0).fillna(0), color="#4C78A8")
    axes[0].set(yticks=np.arange(len(positions)), yticklabels=labels, xlim=(0, 1.05),
                xlabel="Within-position Spearman", title=f"{measure}: rank agreement")
    for index, correlation in enumerate(selected.spearman):
        label = f"{correlation:.3f}" if np.isfinite(correlation) else "N/A"
        axes[0].text(.98, index, label, ha="right", va="center", transform=axes[0].get_yaxis_transform(),
                     bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    draw_ridges(axes[1], values, np.array([group.mean() for group in values]), labels, limits, bins=140, smoothing=5)
    axes[1].set(xlabel=f"Player {measure}: {short_name(left)} minus {short_name(right)}", title="Distribution of score differences")
    outside = int(((delta < limits[0]) | (delta > limits[1])).sum())
    figure.supxlabel(f"Ridges: central 96% range plus padding; equal peak heights. {outside} players outside display remain in every metric.\n"
                     "Negative / undefined correlations are labeled even when no positive bar is drawn.", fontsize=9)
    return figure


def rank_movement_plot(counts, pair, measure, color_max):
    order = counts[["position_bucket", "players"]].drop_duplicates().sort_values(["players", "position_bucket"], ascending=[True, False])
    matrix = counts.pivot(index="position_bucket", columns="rank_delta", values="count").loc[order.position_bucket]
    outside = counts.groupby("position_bucket").outside_display.first().reindex(order.position_bucket)
    figure, axis = plt.subplots(figsize=(14, 6), layout="constrained")
    image = axis.imshow(matrix.to_numpy(), origin="lower", aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=color_max)
    axis.set_xticks(np.arange(len(matrix.columns)), [f"{delta:+d}" if delta else "0" for delta in matrix.columns], fontsize=9)
    axis.set_yticks(np.arange(len(order)), [f"{position} (n={count})" for position, count in zip(order.position_bucket, order.players)])
    for row, values in enumerate(matrix.to_numpy()):
        for column, count in enumerate(values):
            if count:
                axis.text(column, row, str(count), ha="center", va="center", fontsize=8,
                          color="white" if count > color_max / 2 else "black")
        axis.text(1.01, row, f"Outside: {outside.iloc[row]}", transform=axis.get_yaxis_transform(), fontsize=9, va="center")
    axis.set_xticks(np.arange(-.5, len(matrix.columns), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel(f"Position rank: {short_name(pair[0])} minus {short_name(pair[1])}; positive = adapter ranks lower")
    axis.set_title(f"{measure}: exact integer rank changes (one rank per column)")
    figure.colorbar(image, ax=axis, label="Players; shared total/per90 scale", fraction=.04, pad=.16)
    figure.supxlabel(f"{outside.sum()} players outside [{matrix.columns.min()}, +{matrix.columns.max()}] are omitted, not moved to edge bins.\n"
                     "All players remain in ranking metrics.", fontsize=9)
    return figure


def position_scatter_plot(players, pair, measure):
    left, right = pair
    order = players.groupby("position_bucket").size().sort_values(ascending=False)
    values = players[[f"{left}_{measure}", f"{right}_{measure}"]].to_numpy()
    lower, upper = min(0., values.min()), max(0., values.max())
    padding = max(.05 * (upper - lower), .001)
    limits = (lower - padding, upper + padding)
    figure, axes = plt.subplots(int(np.ceil(len(order) / 3)), 3, figsize=(15, 5 * int(np.ceil(len(order) / 3))),
                                sharex=True, sharey=True, squeeze=False, layout="constrained")
    for axis, (position, count) in zip(axes.ravel(), order.items()):
        group = players.loc[players.position_bucket.eq(position)]
        axis.scatter(group[f"{right}_{measure}"], group[f"{left}_{measure}"], s=25, alpha=.8, edgecolors="white", linewidths=.3)
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=.8)
        axis.set(xlim=limits, ylim=limits, title=f"{position} (n={count})")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.15)
    for axis in axes.ravel()[len(order):]:
        axis.set_visible(False)
    figure.supxlabel(f"{LABELS[right]}: {measure}")
    figure.supylabel(f"{LABELS[left]}: {measure}")
    figure.suptitle(f"Signed player {measure} by position; same axes and cohort")
    return figure


