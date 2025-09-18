import numpy as np
import pandas as pd
import datetime
from pandas import Timestamp, DatetimeIndex, DataFrame
from dtcc_solar.utils import SunCollection, SunSkyMapping
from dtcc_solar.utils import SolarParameters, SunPathType
from dtcc_solar.logging import info, debug, warning, error


class Interpolator:

    df_original: DataFrame
    df_reduced: DataFrame
    day_step: int
    min_step: int

    def __init__(self, df_original: DataFrame, day_step: int = 10, min_step: int = 20):
        self.df_original = df_original
        self.day_step = day_step
        self.min_step = min_step
        self.df_reduced = self._interpolate()

    def _interpolate(self):
        """
        Interpolates hourly irradiance data to minute intervals and reduces to representative days.

        Parameters:
        df (DataFrame): Original hourly data with 'dni' and 'dhi'.

        Returns:
        DataFrame: Reduced, interpolated DataFrame conserving total energy.
        """
        df_copy = self.df_original.copy()

        # Just for logging: show what normalization would do to the index
        self._print_normalization_effect(df_copy.index)

        df_interp = self._interp_time(df_copy)
        # Add date column *after* interpolation
        df_interp["date"] = df_interp.index.normalize()
        df_reduced = self._reduce_days(df_interp)

        info(f"Total DNI in original intervals: {np.sum(df_copy['dni']):.2f} Wh/m²")
        info(f"Total DHI in original intervals: {np.sum(df_copy['dhi']):.2f} Wh/m²")
        info(f"Total DNI in interp intervals: {np.sum(df_interp['dni']):.2f} Wh/m²")
        info(f"Total DHI in interp intervals: {np.sum(df_interp['dhi']):.2f} Wh/m²")
        info(f"Total DNI in reduced intervals: {np.sum(df_reduced['dni']):.2f} Wh/m²")
        info(f"Total DHI in reduced intervals: {np.sum(df_reduced['dhi']):.2f} Wh/m²")

        return df_reduced

    def _interp_time(self, df: DataFrame) -> DataFrame:
        """
        Interpolates a DataFrame with hourly DateTimeIndex to t-minute intervals.

        Parameters:
        df (DataFrame): Input DataFrame with hourly steps and 'dni', 'dhi' columns.

        Returns:
        DataFrame: Interpolated DataFrame at t-minute intervals, scaled to Wh per t min.
        """
        t_str = f"{self.min_step}min"

        new_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=t_str)

        df_interp = df.reindex(new_index)
        df_interp[["dni", "dhi"]] = df_interp[["dni", "dhi"]].interpolate(method="time")

        # Convert from Wh/h to Wh per t
        df_interp[["dni", "dhi"]] *= self.min_step / 60.0

        return df_interp

    def _reduce_days(self, df_interp: DataFrame) -> DataFrame:
        """
        Reduces the DataFrame to one representative day per group of `step` days.
        Scales the selected day's `dni` and `dhi` so the total energy is conserved.

        Parameters:
        df_interp (DataFrame): DataFrame with 20-minute intervals and 'date' column.
        step (int): Number of days per group.

        Returns:
        DataFrame: Reduced DataFrame.
        """
        df = df_interp.copy()
        unique_days = sorted(df["date"].unique())
        n_days = len(unique_days)

        reduced_rows = []

        for i in range(0, n_days, self.day_step):
            group_days = unique_days[i : i + self.day_step]
            if len(group_days) == 0:
                continue

            rep_day = group_days[-1]

            group_data = df[df["date"].isin(group_days)]
            rep_data = df[df["date"] == rep_day].copy()

            group_dni_total = group_data["dni"].sum()
            group_dhi_total = group_data["dhi"].sum()

            rep_dni_total = rep_data["dni"].sum()
            rep_dhi_total = rep_data["dhi"].sum()

            dni_factor = group_dni_total / rep_dni_total if rep_dni_total > 0 else 0
            dhi_factor = group_dhi_total / rep_dhi_total if rep_dhi_total > 0 else 0

            rep_data["dni"] *= dni_factor
            rep_data["dhi"] *= dhi_factor

            reduced_rows.append(rep_data)

        df_reduced = pd.concat(reduced_rows)
        df_reduced.drop(columns="date", inplace=True)

        return df_reduced

    def _normalize_datetime_index(self, index: DatetimeIndex) -> DataFrame:

        index_n = index.normalize()

        start_before = index_n[0].replace(tzinfo=None)
        end_before = index_n[-1].replace(tzinfo=None)

        start_after = (start_before.normalize()).replace(tzinfo=None)
        end_after = (end_before.normalize()).replace(tzinfo=None)

        info("Warning: Time intervals have been changed")
        info("--------------------------------------------------------------------")
        info(f"Time period before norm: {start_before} → {end_before}")
        info(f"Time period after norm:  {start_after} → {end_after}")
        info("-------------------------------------------------------------------")

        return index_n

    def _print_normalization_effect(self, index: DatetimeIndex):
        """
        Logs how normalization affects the start and end of a DateTimeIndex.
        """
        start_before = index[0]
        end_before = index[-1]
        start_after = start_before.normalize()
        end_after = end_before.normalize()

        info("Warning: Time intervals are normalized to midnight for day grouping.")
        info("--------------------------------------------------------------------")
        info(f"Time period before normalization: {start_before} → {end_before}")
        info(f"Time period after normalization:  {start_after} → {end_after}")
        info("--------------------------------------------------------------------")
