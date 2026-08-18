from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class TaskLoader:
    """
    A utility class to load, filter, and extract task metadata from a predefined manifest CSV.
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        try:
            self._df = pd.read_csv(manifest_path)
            # Standardize specific columns to string to avoid int/str mismatching
            for col in ["cohort", "rat", "region", "date"]:
                if col in self._df.columns:
                    self._df[col] = self._df[col].astype(str)
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    def filter(
        self,
        rat: Optional[Union[str, int, List[Union[str, int]]]] = None,
        region: Optional[Union[str, List[str]]] = None,
        cohort: Optional[Union[str, List[str]]] = None,
        date: Optional[Union[str, int, List[Union[str, int]]]] = None
    ) -> "TaskLoader":
        """
        Returns a new TaskLoader instance with the filtered subset of data.
        """
        new_loader = TaskLoader.__new__(TaskLoader)
        new_loader.manifest_path = self.manifest_path
        df = self._df.copy()

        if rat is not None:
            rats = [str(rat)] if isinstance(rat, (str, int)) else [str(r) for r in rat]
            df = df[df["rat"].isin(rats)]
        if region is not None:
            regions = [region] if isinstance(region, str) else region
            df = df[df["region"].isin(regions)]
        if cohort is not None:
            cohorts = [cohort] if isinstance(cohort, str) else cohort
            df = df[df["cohort"].isin(cohorts)]
        if date is not None:
            dates = [str(date)] if isinstance(date, (str, int)) else [str(d) for d in date]
            df = df[df["date"].isin(dates)]

        new_loader._df = df
        return new_loader

    @property
    def available_rats(self) -> List[str]:
        return sorted(self._df["rat"].unique().tolist())

    @property
    def available_regions(self) -> List[str]:
        return sorted(self._df["region"].unique().tolist())

    def __len__(self) -> int:
        return len(self._df)

    def to_tasks(self) -> List[Dict]:
        """
        Converts the current internal dataframe into the list-of-dictionaries format
        required by the processing pipelines.
        """
        tasks = []
        for _, row in self._df.iterrows():
            data_path = Path(row["data_path"])
            tasks.append({
                "cohort": row["cohort"],
                "rat": row["rat"],
                "region": row["region"],
                "date": row["date"],
                "data_path": str(data_path),
                "file_name": data_path.name,
                "scoring_path": str(row["scoring_path"]),
            })
        return tasks
