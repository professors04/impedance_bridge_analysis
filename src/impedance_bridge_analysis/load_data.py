
import xarray as xr
from pathlib import Path

from typing import Sequence

import json
import ast


class LoadData():
    """
    A class to load data from Zarr files or QCoDeS databases.

    Attributes:
        data_path (str): The path to the data directory or database file.
    
    The two methods return the data as an xarray Dataset.
    """

    def __init__(self, data_path: str):
        
        self.data_path = data_path
    

    def load_zarr(self, run_id: int) -> xr.Dataset:
        """
        Load data from a Zarr file.

        Returns:
            An xarray Dataset containing the loaded data.
        """

        base_dir = Path(self.data_path)

        matches = [
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(f"{run_id}-") and p.name.endswith(".zarr")
        ]

        xr_dataset = xr.open_zarr(matches[0])

        return xr_dataset
    
    
    def load_database(self, run_id: int) -> xr.Dataset:
        """
        Load data from a QCoDeS database.

        Args:
            run_id (int): The ID of the run to load from the database. Can also be a list.

        Returns:
            An xarray Dataset containing the loaded data.
        """

        initialise_or_create_database_at(self.data_path)

        if isinstance(run_id, Sequence):
            xr_datasets = [load_by_run_spec(captured_run_id=id).to_xarray_dataset() for id in run_id]
            return xr_datasets
        else:
            ds = load_by_run_spec(captured_run_id=run_id)
            xr_dataset = ds.to_xarray_dataset()
            return xr_dataset
    

def get_parameter_snapshot_value(data: xr.Dataset, parameter_name: str):
        
    param_snap = data.attrs["Parameters Snapshot"]
    snapshot_dict = json.loads(param_snap)

    val = snapshot_dict[parameter_name]["value"]

    return ast.literal_eval(val)

