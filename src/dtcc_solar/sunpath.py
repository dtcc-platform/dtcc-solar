import os
import math
import pytz
import datetime
import numpy as np
import pandas as pd
from dtcc_core.model import Mesh, PointCloud
from dtcc_solar import utils
from dtcc_solar.utils import SunCollection, unitize
from dtcc_solar.utils import SolarParameters, SunPathType, SunMapping
from dtcc_solar.interpolate import Interpolator
from pvlib import solarposition
from dtcc_solar.utils import concatenate_meshes, hours_count
from pandas import Timestamp, DatetimeIndex, DataFrame
from dtcc_solar.logging import info, debug, warning, error
from pprint import pp


class Sunpath:
    """
    The Sunpath class represents a sunpath diagram and associated calculations. It
    contains various attributes and methods for calculating sun positions, building
    sunpath meshes, and more.

    Sun positions are stored in the SunCollection object which also contains the
    weather data. There are also two different approaches for approximating sun
    positions with SunGroups or SunQuads.

    Attributes
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    origin : np.ndarray
        Origin point for the sunpath diagram.
    r : float
        Radius of the sunpath diagram should be larger than the model.
    w : float
        Width of paths (anlemmas, daypaths) in the sunpath diagram.
    sunc : SunCollection
        Collection of sun data.
    sun_pc : list[PointCloud]
        List of point clouds representing sun positions.
    sun_pc_minute : PointCloud
        Point cloud of sun positions per minute for a full year.
    mesh : Mesh
        Combination of analemmas and day paths.
    analemmas_meshes : list[Mesh]
        List of meshes for analemmas for each hour in a year.
    daypath_meshes : list[Mesh]
        List of meshes for day paths for three dates in a year.
    analemmas_pc : PointCloud
        Point cloud for analemmas for each hour in a year.
    """

    lat: float
    lon: float
    origin: np.ndarray
    r: float
    w: float

    sunc: SunCollection
    sunc_interp: SunCollection  # Interpolated sun positions for smoother paths
    origin: np.ndarray
    sun_pc: PointCloud
    suns_pc_minute: PointCloud
    mesh: Mesh  # Combination of annalemmas and day paths
    analemmas_meshes: list[Mesh]  # Analemmas for each hour in a year
    daypath_meshes: list[Mesh]  # Day paths for three dates in a year
    analemmas_pc: PointCloud  # Analemmas for each hour in a year as a point cloud
    above_horizon: list[bool]  # List of bools for if sun position is above the horizon

    dt_index: pd.DatetimeIndex  # Datetime index for the DataFrame
    df: pd.DataFrame  # DataFrame with time index and 'dni' and 'dhi' columns
    df_original: pd.DataFrame  # Original DataFrame before interpolation

    def __init__(
        self,
        p: SolarParameters,
        radius: float = 50,
        include_night: bool = False,
        interpolate_df: bool = False,
    ):
        """
        Initializes the Sunpath object with given parameters.

        Parameters
        ----------
        p : SolarParameters
            Solar parameters for sunpath calculations.
        radius : float
            Radius of the sunpath diagram.
        include_night : bool, optional
            Whether to include night time in the sunpath (default is False).
        """
        self.origin = np.array([0, 0, 0])
        self.r = radius
        self.above_horizon = []
        self.sungroups = None
        self.sundome = None
        self.tz_offset = self._get_epw_timezone(p)
        self.dt_index = self._get_datetime_index(p)
        self.df = self._create_irradiance_dataframe(p)

        info("-----------------------------------------------------")
        info("Data retrieved from EPW file:")
        info(f"  Time Zone Offset: {self.tz_offset} hours")
        info(f"  Latitude: {self.lat}°")
        info(f"  Longitude: {self.lon}°")
        info(f"  DataFrame tz: {self.df.index.tz}")
        info(f"  DNI values count: {len(self.df['dni'].values)}")
        info(f"  DHI values count: {len(self.df['dhi'].values)}")
        info("-----------------------------------------------------")

        # Reduce number of solar positions by interpolation
        if interpolate_df:
            interpolator = Interpolator(self.df)
            self.df_original = self.df.copy()
            self.df = interpolator.df_reduced

        self.sunc = self._create_sun_collection(self.df, include_night)

        self.sunc.info_print()

        if p.display:
            self._create_sunpath_mesh()

    # Create time stamps and retrieve irradiance data from EPW file

    def _get_epw_timezone(self, p: SolarParameters) -> int:
        """
        Extracts the 'Time Zone' offset from the EPW file header.

        Returns:
            int: The time zone offset in hours from UTC (e.g., +1, -5).
        """

        labels = [
            "Header label   ",
            "Station name   ",
            "State/Province ",
            "Country        ",
            "Data type      ",
            "WMO station ID ",
            "Latitude       ",
            "Longitude      ",
            "Time Zone      ",
            "Elevation (m)  ",
        ]

        with open(p.weather_file, "r") as file:
            first_line = file.readline()
            parts = first_line.strip().split(",")
            info("-----------------------------------------------------")
            info("From EPW header:")
            for i, part in enumerate(parts):
                label = "Unknown label  "
                if i < len(labels):
                    label = labels[i]
                info(f"  {label}: {part}")
            info("-----------------------------------------------------")
            try:
                # According to EPW format, field 8 is the time zone (index 7)
                latitude = float(parts[6])
                longitude = float(parts[7])
                tz_offset = int(float(parts[8]))

            except (IndexError, ValueError) as e:
                raise ValueError("Could not parse time zone from EPW header.") from e

        self.lat = latitude
        self.lon = longitude

        return tz_offset

    def _get_datetime_index(self, p: SolarParameters) -> DatetimeIndex:
        """
        Generate correctly aligned datetime index for EPW data (start of hour, local standard time).
        """

        n_hours = hours_count(p.start, p.end)

        if n_hours <= 0:
            error("Start date must be before end date.")
            raise ValueError("Invalid date range for solar analysis.")
        elif n_hours > 8760:
            warning("More than 8760 hours, -> leap year or incorrect date range.")
            n_hours = 8760

        # EPW timestamps are in local STANDARD time (no DST)
        index = pd.date_range(start=p.start, periods=n_hours, freq="h")

        # Invert sign for Etc/GMT
        index = index.tz_localize(f"Etc/GMT{-self.tz_offset}")

        return index

    def _create_irradiance_dataframe(self, p: SolarParameters) -> DataFrame:
        """
        Extract DNI and DHI data from an EPW file and align it with the given datetime index.

        Parameters:
            p (SolarParameters): Object with .weather_file path.
            datetime_index (DatetimeIndex): Timezone-localized index matching EPW time format.

        Returns:
            pd.DataFrame: A DataFrame with 'dni' and 'dhi' columns aligned to the datetime index.
        """

        # Step 1: Load the EPW data (skip 8 header rows)
        df = pd.read_csv(p.weather_file, skiprows=8, header=None)

        # Step 2: Create the full EPW datetime index (EPW timestamps are END of hour)
        full_index_endhour = pd.date_range(
            start="2019-01-01 00:00",  # Adjust if not 2019 or make it dynamic
            periods=len(df),
            freq="h",
            tz=f"Etc/GMT{-self.tz_offset}",  # EPW time is in local standard time
        )

        full_index_starthour = pd.date_range(
            start="2019-01-01 00:00",  # Adjust if not 2019 or make it dynamic
            periods=len(df),
            freq="h",
            tz=f"Etc/GMT{-self.tz_offset}",  # EPW time is in local standard time
        )

        # Step 3: Assign full index to the DataFrame
        df.index = full_index_starthour

        # Step 4: Extract DNI and DHI columns (0-indexed)
        DNI_COL = 14  # Column 15
        DHI_COL = 15  # Column 16
        irradiance_df = df[[DNI_COL, DHI_COL]]
        irradiance_df.columns = ["dni", "dhi"]

        # Step 5: Reindex to match requested times
        result_df = irradiance_df.reindex(self.dt_index)

        # Step 6: Fill NaN values with 0.0
        result_df = result_df.fillna(0.0)

        return result_df

    def _create_sun_collection(self, df: DataFrame, include_night: bool = False):
        """
        Create sun positions and append weather data.

        Parameters
        ----------
        p : SolarParameters
            Parameters for solar calculations.
        include_night : bool, optional
            Flag to include night times in the sunpath (default is False).
        """
        sunc = SunCollection()
        sunc.time_stamps = list(pd.to_datetime(df.index))
        sunc.count = len(sunc.time_stamps)
        sunc.count_below = 0
        sunc.count_above = sunc.count
        self._calc_suns_positions(sunc, df)

        if not include_night:
            self._remove_suns_below_horizon(sunc)

        remain = sunc.count - sunc.count_below
        info("-----------------------------------------------------")
        info("Sun Collection created from data frame:")
        info(f"  Date range resulting in {sunc.count} total sun positions")
        info(f"  Removed {sunc.count_below} suns that were below the horizon.")
        info(f"  Remaining suns for analysis: {remain}")
        info("-----------------------------------------------------")

        return sunc

    def _calc_suns_positions(self, sunc: SunCollection, df: DataFrame):
        """
        Calculate sun positions for the timestamps in provided DataFrame.
        """
        solpos = solarposition.get_solarposition(df.index, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        zeni = np.radians(solpos.apparent_zenith.to_list())
        x_sun = self.r * np.cos(elev) * np.sin(-azim) + self.origin[0]
        y_sun = self.r * np.cos(elev) * np.cos(-azim) + self.origin[1]
        z_sun = self.r * np.sin(elev) + self.origin[2]
        sun_vecs = []
        positions = []
        zeniths = []
        for i in range(sunc.count):
            sun_vec = unitize(np.array([x_sun[i], y_sun[i], z_sun[i]]))
            sun_vecs.append(sun_vec)
            positions.append(np.array([x_sun[i], y_sun[i], z_sun[i]]))
            zeniths.append(zeni[i])

        sunc.sun_vecs = np.array(sun_vecs)
        sunc.positions = np.array(positions)
        sunc.zeniths = np.array(zeniths)
        sunc.dni = df["dni"].to_numpy()
        sunc.dhi = df["dhi"].to_numpy()

    def _remove_suns_below_horizon(self, sunc: SunCollection):
        """
        Remove sun positions below the horizon.
        """
        above = sunc.zeniths < (math.pi / 2.0)
        sunc.count_above = np.sum(np.array(above, dtype=np.int32))
        sunc.count_below = sunc.count - sunc.count_above

        sunc.sun_vecs = sunc.sun_vecs[above, :]
        sunc.positions = sunc.positions[above, :]
        sunc.dni = sunc.dni[above]
        sunc.dhi = sunc.dhi[above]
        sunc.zeniths = sunc.zeniths[above]

        filtered_timestamps = []
        for ts, valid in zip(sunc.time_stamps, above):
            if valid:
                filtered_timestamps.append(ts)

        sunc.time_stamps = filtered_timestamps
        sunc.count = len(sunc.time_stamps)

    # Create 3D geometry for sunpath visualization.

    def _create_sunpath_mesh(self):
        """Build sunpath mesh by combining analemmas and day paths."""
        self.w = self.r / 300

        # Get analemmas mesh and sun positions represented as a point cloud
        sun_pos_dict = self._calc_analemmas(2019, 2)
        self.analemmas_meshes = self._create_mesh_strip(sun_pos_dict, self.w)
        self.analemmas_pc = self._create_sunpath_pc(sun_pos_dict)

        # Get every sun position for each minute in a year as points in a point cloud
        dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="1D")
        sun_pos_dict = self._calc_daypaths(dates, 1)
        self.suns_pc_minute = self._create_sunpath_pc(sun_pos_dict)

        # Get day path loops as mesh strips
        days = pd.to_datetime(["2019-06-21", "2019-03-21", "2019-12-21"])
        days_dict = self._calc_daypaths(days, 2)
        self.daypath_meshes = self._create_mesh_strip(days_dict, self.w)

        # Create pc representation of current sun positions
        self.sun_pc = self._create_sun_spheres()

        mesh1 = concatenate_meshes(self.analemmas_meshes)
        mesh2 = concatenate_meshes(self.daypath_meshes)
        self.mesh = concatenate_meshes([mesh1, mesh2])

        info("-----------------------------------------------------")
        info("Sunpath geometry created for visualisation:")
        info(f"  Analemmas: {len(self.analemmas_meshes)} meshes")
        info(f"  Day paths: {len(self.daypath_meshes)} meshes")
        info(f"  Sun count in visualisation point cloud: {len(self.sun_pc.points)}")
        info("-----------------------------------------------------")

    def _calc_analemmas(self, year: int, sample_rate: int):
        """
        Calculates analemmas for a given year and sample rate.

        Parameters
        ----------
        year : int
            Year for which to calculate analemmas.
        sample_rate : int
            Rate at which to sample sun positions.

        Returns
        -------
        dict
            Dictionary of sun positions for each hour.
        """
        start_date = str(year) + "-01-01 12:00:00"
        end_date = str(year + 1) + "-01-01 11:00:00"
        times = pd.date_range(start=start_date, end=end_date, freq="h")

        # Times to evaluate will be 8760 for a normal year and 8764 for leep year
        times_to_evaluate = len(times.values)

        # Number of days will be 365 or 366 at leep year
        num_of_days = int(np.ceil(times_to_evaluate / 24))

        # Get solar position from the pvLib libra
        sol_pos_hour = solarposition.get_solarposition(times, self.lat, self.lon)

        # Reduce the number of days to reduce the density of the sunpath diagram
        days = np.zeros(num_of_days)
        num_evaluated_days = len(days[0 : len(days) : sample_rate])

        elev = np.zeros((24, num_evaluated_days))
        azim = np.zeros((24, num_evaluated_days))
        zeni = np.zeros((24, num_evaluated_days))

        loop_hours = np.unique(sol_pos_hour.index.hour)
        sun_pos_dict = dict.fromkeys([h for h in loop_hours])

        r = self.r

        # Get hourly sun path loops in matrix form and elevaion, azimuth and zenith coordinates
        for h in loop_hours:
            subset = sol_pos_hour.loc[sol_pos_hour.index.hour == h, :]
            rad_elev = np.radians(subset.elevation)
            rad_azim = np.radians(subset.azimuth)
            rad_zeni = np.radians(subset.zenith)
            elev[h, :] = rad_elev.values[0 : len(rad_elev.values) : sample_rate]
            azim[h, :] = rad_azim.values[0 : len(rad_azim.values) : sample_rate]
            zeni[h, :] = rad_zeni.values[0 : len(rad_zeni.values) : sample_rate]

        # Convert hourly sun path loops from spherical to cartesian coordinates
        for h in range(0, 24):
            x = r * np.cos(elev[h, :]) * np.sin(-azim[h, :]) + self.origin[0]
            y = r * np.cos(elev[h, :]) * np.cos(-azim[h, :]) + self.origin[1]
            z = r * np.sin(elev[h, :]) + self.origin[2]

            sun_pos_dict[h] = utils.create_list_of_vec3(x, y, z)

        return sun_pos_dict

    def _calc_daypaths(self, dates: pd.DatetimeIndex, minute_step: float):
        """
        Calculate sun positions for day paths used to visualise days.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates for the calculation.
        minute_step : float
            Time step in minutes for the calculation.

        Returns
        -------
        dict
            Dictionary of sun positions for each day.
        """
        n = len(dates.values)
        n_evaluation = int(math.ceil(24 * 60 / minute_step))
        elev = np.zeros((n, n_evaluation + 1))
        azim = np.zeros((n, n_evaluation + 1))
        date_counter = 0

        days_dict = dict.fromkeys([d for d in range(0, n)])
        min_step_str = str(minute_step) + "min"
        day_step = pd.Timedelta(str(24) + "h")

        for date in dates.values:
            times = pd.date_range(date, date + day_step, freq=min_step_str)
            sol_pos_day = solarposition.get_solarposition(times, self.lat, self.lon)
            rad_elev_day = np.radians(sol_pos_day.elevation)
            rad_azim_day = np.radians(sol_pos_day.azimuth)
            elev[date_counter, :] = rad_elev_day.values
            azim[date_counter, :] = rad_azim_day.values
            date_counter = date_counter + 1

        for d in range(0, date_counter):
            x = self.r * np.cos(elev[d, :]) * np.sin(-azim[d, :]) + self.origin[0]
            y = self.r * np.cos(elev[d, :]) * np.cos(-azim[d, :]) + self.origin[1]
            z = self.r * np.sin(elev[d, :]) + self.origin[2]

            days_dict[d] = utils.create_list_of_vec3(x, y, z)

        return days_dict

    def _create_mesh_strip(self, sun_pos_dict, width: float):
        """
        Create mesh strip for visualizing analemmas

        Parameters
        ----------
        sun_pos_dict : Dict
            Dictionary with sunpositions for each hour from 0 - 23.
        width : float
            Width of the mesh strip.
        """
        meshes = []

        for h in sun_pos_dict:
            n_suns = len(sun_pos_dict[h])
            pos = sun_pos_dict[h]
            offset_vertices_L = np.zeros((n_suns + 1, 3))
            offset_vertices_R = np.zeros((n_suns + 1, 3))

            v_counter = 0
            vertices = []
            faces = []

            for i in range(0, n_suns - 1):
                i_next = i + 1
                if i == n_suns - 1:
                    i_next = 0

                sun_pos_1 = np.array([pos[i].x, pos[i].y, pos[i].z])
                sun_pos_2 = np.array([pos[i_next].x, pos[i_next].y, pos[i_next].z])

                vec_1 = unitize(0.5 * (sun_pos_1 + sun_pos_2))
                vec_2 = unitize(sun_pos_2 - sun_pos_1)
                vec_3 = np.cross(vec_1, vec_2)

                # Offset vertices to create a width for path
                offset_vertices_L[i, :] = sun_pos_1 + (width * vec_3)
                offset_vertices_R[i, :] = sun_pos_1 - (width * vec_3)

                vertices.append(offset_vertices_L[i, :])
                vertices.append(offset_vertices_R[i, :])

                if i < (n_suns - 2):
                    faces.append([v_counter, v_counter + 2, v_counter + 1])
                    faces.append([v_counter + 1, v_counter + 2, v_counter + 3])
                    v_counter += 2
                else:
                    faces.append([v_counter, 0, v_counter + 1])
                    faces.append([v_counter + 1, 0, 1])

            vertices = np.array(vertices)
            faces = np.array(faces)
            mesh = Mesh(vertices=vertices, faces=faces)
            meshes.append(mesh)

        return meshes

    def _create_sun_spheres(self):
        """Create point clouds for sun positions."""
        points = []
        points = self.sunc.positions
        points = np.array(points)
        pc = PointCloud(points=points)
        return pc

    def _create_sunpath_pc(self, sun_pos_dict):
        """Create point cloud representation of sun positions."""
        points = []
        for h in sun_pos_dict:
            pos_list = sun_pos_dict[h]
            for pos in pos_list:
                sun_pos = np.array([pos.x, pos.y, pos.z])
                points.append(sun_pos)

        points = np.array(points)
        pc = PointCloud(points=points)
        return pc
