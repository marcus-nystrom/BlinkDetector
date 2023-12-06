# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:56:52 2022

@author: Marcus
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import blink


# Import and change (optional) settings
settings = blink.Settings()
settings.plot_on = False

bd = blink.BlinkDetector(settings)

eyes = ['left', 'right']

dataset = 'Spectrum'
cwd = Path.cwd()
path_spectrum = cwd / 'data' / 'spectrum'
path_fusion = cwd / 'data' / 'fusion'

# %% Run classification for the 12 participants recorded in the DC
if 'Spectrum' in dataset:

    # list all folders (each folder is a participant) in trial
    pids = [f for f in path_spectrum.iterdir() if f.is_dir()] #[4]

    out_eo = []
    out_pupil = []

    for pid in pids:
        for eye in eyes:
            key = f'{eye}_eye_openness_value'

            pid_name = str(pid).split(os.sep)[-1]
            files = pid.rglob('*.tsv')
            for file in files:

                filename = str(file).split(os.sep)[-1][:-4]

                # if 'Center' not in filename:
                #     continue

                print(f'pid: {pid_name}, eye: {eye}, condition: {filename}')
                df = pd.read_csv(Path(file), sep='\t')
                eye_openness_signal = np.c_[df[key]]
                eye_openness_signal = np.squeeze(eye_openness_signal)

                pupil_signal = np.array(df[f'{eye}_pupil_diameter'])
                t = np.array(df['system_time_stamp'])
                t = (t - t[0]) / 1000

                nan_pupil = np.sum(np.isnan(pupil_signal)) / len(pupil_signal)
                nan_eyeopenness = np.sum(np.isnan(eye_openness_signal)) / len(eye_openness_signal)

                xy = np.c_[df[f'{eye}_gaze_point_on_display_area_x'],
                            df[f'{eye}_gaze_point_on_display_area_y']]

                df_out, eye_openness_signal_vel = bd.blink_detector_eo(t, eye_openness_signal, settings.Fs, filter_length=settings.filter_length,
                                                                 gap_dur=settings.gap_dur,
                                                                 width_of_blink=settings.width_of_blink,
                                                                 min_separation=settings.min_separation)

                df_out_pupil = bd.blink_detector_pupil(t, pupil_signal, settings.Fs,
                                                           gap_dur=settings.gap_dur,
                                                           min_dur=settings.min_blink_dur,
                                                           min_separation=settings.min_separation)
                if settings.plot_on:
                    bd.plot_blink_detection_results(t,
                                                 eye_openness_signal,
                                                 eye_openness_signal_vel,
                                                 df_out,
                                                 pid_name,
                                                 filename,
                                                 eye,
                                                 pupil_signal=pupil_signal,
                                                 df_blink_pupil = df_out_pupil,
                                                 xy = xy)

                # Add participant ID to data frame
                df_out['pid'] = pid_name
                df_out['eye'] = eye
                df_out['trial'] = filename
                df_out['trial_duration'] = len(eye_openness_signal) / settings.Fs
                df_out['blink_rate'] = len(df_out) / df_out['trial_duration']
                out_eo.append(df_out)

                df_out_pupil['pid'] = pid_name
                df_out_pupil['eye'] = eye
                df_out_pupil['trial'] = filename
                df_out_pupil['trial_duration'] = len(pupil_signal) / settings.Fs
                df_out_pupil['blink_rate'] = len(df_out_pupil) / df_out_pupil['trial_duration']
                out_pupil.append(df_out_pupil)

    df_all_eo = pd.concat(out_eo)
    df_all_pupil = pd.concat(out_pupil)

    df_all_eo.to_csv('eo.csv')
    df_all_pupil.to_csv('pupil.csv')


# %% Run classification for Fusion trials
else:

    settings.Fs = 120
    out_eo = []
    out_pupil = []
    for eye in eyes:
        key = f'Eye openness {eye}'

        files = path_fusion.rglob('*.tsv')
        for file in files:

            filename = str(file).split(os.sep)[-1][:-4]
            pid_name = filename.split('-')[0].strip()
            filename = filename.split('-')[1].strip()

            print(file)
            df = pd.read_csv(Path(file), sep='\t', decimal = ',')
            eye_openness_signal = np.c_[df[key]]
            eye_openness_signal = np.squeeze(eye_openness_signal)

            pupil_signal = np.array(df[f'Pupil diameter {eye}'])
            t = np.array(df['Recording timestamp'])
            t = (t - t[0]) / 1000

            nan_pupil = np.sum(np.isnan(pupil_signal)) / len(pupil_signal)
            nan_eyeopenness = np.sum(np.isnan(eye_openness_signal)) / len(eye_openness_signal)

            xy = np.c_[df[f'Gaze direction {eye} X'],
                        df[f'Gaze direction {eye} Y']]

            df_out, eye_openness_signal_vel = bd.blink_detector_eo(t, eye_openness_signal, settings.Fs, filter_length=settings.filter_length,
                                                             gap_dur=settings.gap_dur,
                                                             width_of_blink=settings.width_of_blink,
                                                             min_separation=settings.min_separation)

            df_out_pupil = bd.blink_detector_pupil(t, pupil_signal, settings.Fs,
                                                       gap_dur=settings.gap_dur,
                                                       min_dur=settings.min_blink_dur,
                                                       min_separation=settings.min_separation)
            if settings.plot_on:
                bd.plot_blink_detection_results(t,
                                             eye_openness_signal,
                                             eye_openness_signal_vel,
                                             df_out,
                                             pid_name,
                                             filename,
                                             eye,
                                             pupil_signal=pupil_signal,
                                             df_blink_pupil = df_out_pupil,
                                             xy = xy)