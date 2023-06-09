#!/usr/bin/env python3
from dataclasses import dataclass
import pandas as pd
import matplotlib as mpl
from operator import itemgetter
from typing import Any
import re
import datetime
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def read_binary_data(filename: str) -> pd.DataFrame:
    dt = np.dtype(
        [
            ("min_loss", "f4"),
            ("max_loss", "f4"),
            ("mean_loss", "f4"),
            ("variance", "f4"),
        ]
    )

    data = np.fromfile(filename, dtype=dt)
    return pd.DataFrame(data, columns=data.dtype.names)  # type: ignore


def find_the_most_recent_results_dir(parent: Path):
    candidates = list(filter(lambda p: p.is_dir(), parent.iterdir()))

    if not candidates:
        raise RuntimeError(f"no result directories at location: {parent}")

    dates = []

    pattern = re.compile(
        r"(?P<day>\d\d)-(?P<month>\d\d)-(?P<year>\d{4})_(?P<hour>\d\d)-(?P<minute>\d\d)-(?P<second>\d\d)"
    )

    for c in candidates:
        match = pattern.match(c.name)

        if match is None:
            print(f"ignoring directory: {c}")
            continue

        groups = match.groupdict()
        dates.append(
            datetime.datetime(
                **{
                    name: int(groups[name])
                    for name in ("year", "month", "day", "hour", "minute", "second")
                }
            )
        )

    if not dates:
        raise RuntimeError(f"no result directories at location: {parent}")

    max_idx, _ = max(enumerate(dates), key=itemgetter(1))

    return candidates[max_idx]


@dataclass
class ConfigFileSyntaxError(RuntimeError):
    path: Path

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.path}"


def parse_parameters(results_path: Path) -> dict[str, Any] | None:
    dirpath = results_path if results_path.is_dir() else results_path.parent
    params_file = dirpath / "parameters.txt"

    if not params_file.exists():
        return None

    mapped = {}

    with open(params_file, "r") as f:
        for line in f:
            try:
                param_name, value = line.split()
                param_name = param_name.strip(":")
                value = value.strip(",")

                param_type = float if "." in value else int
                mapped[param_name] = param_type(value)
            except ValueError as e:
                raise ConfigFileSyntaxError(params_file) from e

    return mapped


def mark_epochs(params: dict[str, Any], ax: mpl.axis.Axis):
    n = params["iterationsPerEpoch"]
    xs = ax.get_lines()[0].get_xdata()
    epoch_xs = xs[::n]

    style = {"linestyle": "dotted", "color": "#333333", "alpha": 0.5}

    line = None

    for x in epoch_xs:
        line = ax.axvline(x, **style)

    if line is not None:
        line.label = "epoch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to file with algorithm results or to directory with "
            "multiple results."
        ),
    )
    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help=(
            "if called on directory with multiple results directories, "
            " take the most recent"
        ),
    )
    parser.add_argument("-E", "--no-mark-epochs", action="store_true")
    parser.add_argument("-p", "--plot", type=bool, default=True)
    args = parser.parse_args()

    readpath = args.path

    if not readpath.exists():
        raise RuntimeError(f"directory does not exist: {readpath}")

    if readpath.is_dir():
        if args.latest:
            # set the readpath to the most recent directory
            readpath = find_the_most_recent_results_dir(readpath)
            print("most recent results:", readpath)

        # show all island results in the directory
        readfiles = list(
            filter(lambda f: f.is_file() and f.suffix == ".dat", readpath.iterdir())
        )
        readfiles.sort()
    else:
        readfiles = [readpath]

    for readfile in readfiles:
        data = read_binary_data(readfile)

        # mean +- stddevreadfile
        std = np.sqrt(data.variance)
        std_low_line = data.mean_loss - std
        std_high_line = data.mean_loss + std

        x = np.arange(len(data.index))

        min_loss_x = data.min_loss.argmin()
        min_loss_y = data.min_loss[min_loss_x]

        marker = "."

        sns.set_style("whitegrid")
        fig = plt.figure()
        ax = fig.gca()

        sns.lineplot(
            data=data,
            x=x,
            y="mean_loss",
            label=r"$\mu$",
            color="#0000ff",
            marker=marker,
        )
        sns.lineplot(
            data=data, x=x, y="min_loss", label="min", color="#00ff00", marker=marker
        )
        sns.lineplot(
            data=data, x=x, y="max_loss", label="max", color="#ff0000", marker=marker
        )

        # ax = fig.gca()
        # lines = ax.get_lines()

        # ax.fill_between(
        #     x, lines[1].get_ydata(), lines[2].get_ydata(), color="#4444ff", alpha=0.5
        # )

        # plot mean +- stddev
        stddev_color = "#ff9900"
        stddev_linestyle = "--"
        sns.lineplot(
            x=x,
            y=std_low_line,
            label=r"$\mu \pm \sigma$",
            color=stddev_color,
            linestyle=stddev_linestyle,
            marker=marker,
        )
        sns.lineplot(
            x=x,
            y=std_high_line,
            color=stddev_color,
            linestyle=stddev_linestyle,
            marker=marker,
        )

        # fill between mean +- stddev
        # lines = ax.get_lines()
        # ax.fill_between(
        #     x, lines[3].get_ydata(), lines[4].get_ydata(), color="#ff4444", alpha=0.5
        # )

        ax.set_title(readfile.name)
        ax.set_ylabel("loss function statistics")
        ax.set_xlabel("iterations")

        # mark minimum with lines
        min_markline_style = {"linestyle": "--", "color": "gray"}
        ax.axhline(min_loss_y, **min_markline_style)
        ax.axvline(min_loss_x, **min_markline_style)
        ax.annotate(f"{min_loss_y:.2f}, i: {min_loss_x}", (min_loss_x, min_loss_y))

        if not args.no_mark_epochs:
            params = parse_parameters(readfile)

            if params:
                mark_epochs(params, ax)

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        if args.plot:
            plt.show()

        savepath = readfile.parent / (readfile.stem + "_plot.png")
        print(f"saving plot at: {savepath}")
        fig.savefig(savepath)  # type: ignore


if __name__ == "__main__":
    main()
