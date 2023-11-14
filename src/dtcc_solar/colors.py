import numpy as np
from dtcc_solar.utils import ColorBy, OutputCollection
from pprint import pp


def create_data_dict(outc: OutputCollection):
    fsa = np.sum(outc.face_sun_angles, axis=0)
    occ = np.sum((1 - outc.occlusion), axis=0)
    dn = np.sum(outc.irradiance_dn, axis=0)
    # di = np.sum(outc.irradiance_di, axis=0)
    # dh = np.sum(outc.irradiance_dh, axis=0)
    data_dict = {
        "face sun angles": fsa,
        "inverse occlusion": occ,
        "direct normal irradiance": dn,
    }

    return data_dict


def color_mesh(outc: OutputCollection, color_by: ColorBy):
    colors = []
    if color_by == ColorBy.face_sun_angle:
        fsa = np.sum(outc.face_sun_angles, axis=0)
        colors = calc_colors(fsa)
    elif color_by == ColorBy.occlusion:
        occ = np.sum(outc.occlusion, axis=0)
        colors = calc_colors(occ)
    elif color_by == ColorBy.irradiance_dn:
        dn = np.sum(outc.irradiance_dn, axis=0)
        colors = calc_colors(dn)
    elif color_by == ColorBy.irradiance_dh:
        dh = np.sum(outc.irradiance_dh, axis=0)
        colors = calc_colors(dh)
    elif color_by == ColorBy.irradiance_di:
        di = np.sum(outc.irradiance_di, axis=0)
        colors = calc_colors(di)
    elif color_by == ColorBy.irradiance_tot:
        pp(outc.irradiance_dn)
        dn = np.sum(outc.irradiance_dn, axis=0)
        di = np.sum(outc.irradiance_di, axis=0)
        pp(dn)
        tot = dn + di
        colors = calc_colors(tot)
    else:
        print("Color calculation for city mesh failed!")

    return colors


def color_analemmas():
    pass


def calc_colors(values):
    colors = []
    values = np.array(values, dtype=float)
    min_value = np.min(values)
    max_value = np.max(values)
    for i in range(0, len(values)):
        c = get_blended_color(min_value, max_value, values[i])
        colors.append(c)
    return np.array(colors)


# Calculate color blend for a range of values where some are excluded using a True-False mask
def calc_colors_with_mask(values, mask):
    colors = []
    min = np.min(values[mask])
    max = np.max(values[mask])
    for i in range(0, len(values)):
        if mask[i]:
            c = get_blended_color(min, max, values[i])
        else:
            c = [0.2, 0.2, 0.2, 1]
        colors.append(c)
    return colors


# Calculate bleded color in a monochrome scale
def calc_colors_mono(values):
    colors = []
    max_value = np.max(values)
    for i in range(0, len(values)):
        fColor = get_blended_color_mono(max_value, values[i])
        colors.append(fColor)


def get_blended_color(min, max, value):
    diff = max - min
    newMax = diff
    newValue = value - min
    percentage = 100.0 * (newValue / newMax)

    if percentage >= 0.0 and percentage <= 25.0:
        # Blue fading to Cyan [0,x,255], where x is increasing from 0 to 255
        frac = percentage / 25.0
        return [0.0, (frac * 1.0), 1.0]

    elif percentage > 25.0 and percentage <= 50.0:
        # Cyan fading to Green [0,255,x], where x is decreasing from 255 to 0
        frac = 1.0 - abs(percentage - 25.0) / 25.0
        return [0.0, 1.0, (frac * 1.0)]

    elif percentage > 50.0 and percentage <= 75.0:
        # Green fading to Yellow [x,255,0], where x is increasing from 0 to 255
        frac = abs(percentage - 50.0) / 25.0
        return [(frac * 1.0), 1.0, 0.0]

    elif percentage > 75.0 and percentage <= 100.0:
        # Yellow fading to red [255,x,0], where x is decreasing from 255 to 0
        frac = 1.0 - abs(percentage - 75.0) / 25.0
        return [1.0, (frac * 1.0), 0.0]

    elif percentage > 100.0:
        # Returning red if the value overshoot the limit.
        return [1.0, 0.0, 0.0]

    return [0.5, 0.5, 0.5]


def get_blended_color_mono(max, value):
    frac = 0
    if max > 0:
        frac = value / max
    return [frac, frac, frac]


def get_blended_color_red_blue(max, value):
    frac = 0
    if max > 0:
        frac = value / max
    return [frac, 0.0, 1 - frac]


def get_blended_color_yellow_red(max, value):
    percentage = 100.0 * (value / max)
    if value < 0:
        return [1.0, 1.0, 1.0]
    else:
        # Yellow [255, 255, 0] fading to red [255, 0, 0]
        frac = 1 - percentage / 100
        return [1.0, (frac * 1.0), 0.0]
