#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--plot", type=bool, default=True)
    namespace = parser.parse_args()

    readpath = namespace.path

    if readpath.is_dir():
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

        marker = "o"

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

        if namespace.plot:
            plt.show()

        savepath = readfile.parent / (readfile.stem + "_plot.png")
        print(f"saving plot at: {savepath}")
        fig.savefig(savepath)


if __name__ == "__main__":
    main()
