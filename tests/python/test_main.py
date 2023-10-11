from dtcc_solar.scripts.main import run_script


class TestSolar:
    inputfile_S: str
    inputfile_M: str
    inputfile_L: str

    weather_file_clm: str
    weather_file_epw: str

    def setup_method(self):
        self.inputfile_S = "../data/models/CitySurfaceS.stl"
        self.inputfile_M = "../data/models/CitySurfaceM.stl"
        self.inputfile_L = "../data/models/CitySurfaceL.stl"

        self.weather_file_clm = (
            "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        )
        self.weather_file_epw = (
            "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
        )

    ########################### Testig of instant analysis ##############################
    def test_instant_solar_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "1",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 09:00:00",
                "--end_date",
                "2019-03-30 09:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )

    def test_instant_sky_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "2",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 09:00:00",
                "--end_date",
                "2019-03-30 09:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )

    def test_instant_com_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "3",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 09:00:00",
                "--end_date",
                "2019-03-30 09:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )

    ########################################################################################

    ########################### Testing diffusion calculations ##############################
    def test_iterative_sun_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "1",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 10:00:00",
                "--end_date",
                "2019-03-30 15:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )

    def test_iterative_sky_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "2",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 10:00:00",
                "--end_date",
                "2019-03-30 15:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )

    # Iterative testing
    def test_iterative_com_raycasting(self):
        assert run_script(
            [
                "--inputfile",
                self.inputfile_S,
                "--analysis",
                "3",
                "--prep_disp",
                "0",
                "--display",
                "0",
                "--start_date",
                "2019-03-30 10:00:00",
                "--end_date",
                "2019-03-30 15:00:00",
                "--data_source",
                "3",
                "--w_file",
                self.weather_file_clm,
            ]
        )
