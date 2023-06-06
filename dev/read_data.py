#!/usr/bin/env/python3
import pandas as pd
import numpy as np
import argparse

def read_binary_data(filename: str) -> pd.DataFrame:
    dt = np.dtype([
        ("min_loss", "f4"), ("max_loss", "f4"),
        ("mean_loss", "f4"), ("variance", "f4")])

    data = np.fromfile(filename, dtype=dt)
    return pd.DataFrame(data, columns=data.dtype.names) # type: ignore

def get_readfile():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str)
    namespace = parser.parse_args()
    return namespace.dirname


def main():
    readfile = get_readfile()
    data = read_binary_data(readfile)
    print(data)
    # sns.lineplot(data)

if __name__ == "__main__":
    main()
