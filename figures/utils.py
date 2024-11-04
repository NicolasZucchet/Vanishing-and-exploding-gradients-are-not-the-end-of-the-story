import glob
import os
from pathlib import Path

import pandas as pd
import wandb


def load_sweep(sweep_id, last_step_only=False):
    file_name = "cache/{}.csv".format(sweep_id.replace("/", "_"))

    if len(glob.glob(file_name)) > 0:
        return pd.read_csv(file_name)
    else:
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        result = parse_sweep_to_dataframe(sweep_id, last_step_only=last_step_only)
        result.to_csv(file_name)
        return result


def parse_sweep_to_dataframe(sweep_id, last_step_only=False):
    df_list = []
    for i, run in enumerate(wandb.Api().sweep(sweep_id).runs):
        # Skip crashed runs
        if run.state == "crashed":
            print("Skipping crashed {}".format(run.name))
            continue
        else:
            print("Downloading {}".format(run.name))

        # Download the full history of all metrics excluding objects (histograms, etc.)
        df = run.history(x_axis="_step", pandas=True, stream="default")
        df = df.select_dtypes(exclude=["object"])

        if last_step_only:
            df = df[df["_step"] == df["_step"].max()].reset_index()

        # Add summary entries to datframe
        for column_name, value in run.summary.items():
            if column_name not in df.columns and (
                isinstance(value, int) or isinstance(value, float)
            ):
                df[column_name] = value

        # Add config entries to dataframe
        for k, v in run.config.items():
            df[k] = pd.Series(len(df.index) * [v])

        df_list.append(df)

    return pd.concat(df_list)
