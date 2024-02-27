# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:06:07 2023

@author: Marcus

"""

import numpy as np
import preprocessing
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt


# %%
class Settings():
    def __init__(self):

        self.plot_on = False # Visualize detection results
        self.save_fig = False # Save visualization to file
        self.debug = False # Prints output when debugging

        self.Fs = 600  # Sample rate of eye tracker

        # in mm, used to distinguish full from partial blinks
        # (only for visualization purposes)
        self.full_blink_max_opening = 2

        self.gap_dur = 40 # max gaps between period of data loss, interpolate smaller gaps
        self.min_amplitude = 0.1 # % of fully open eye (0.1 - 10%)
        self.min_separation = 100 # min separation between blinks

        self.filter_length = 25  # in ms
        self.width_of_blink = 15  # in ms width of peak to initially detect
        self.min_blink_dur = 30  # reject blinks shorter than 30 ms

        self.min_pupil_size = 2 # in mm
        self.window_len = np.nan # in ms window over which to exclude outliers (np.nan means whole trial)
        self.treshold_SD = 2.5 # remove values 2.5 * SD from the mean


# %%
class BlinkDetector(object):

    def __init__(self, settings):
            '''
            Constructs an instance of the BlinkDetector class, with specified settings
            acquited through the call (get_defaults)
            '''

            self.settings = settings
    # %%
    def _binary2bounds(self, binary):
        """

        Args:
            binary (binary array): 1 represents potential blinks.

        Returns:
            onsets (list): blink onsets (sample idx)
            offsets (list): blink offsets  (sample idx)

        """

        # Find segments of data loss
        d = np.diff(binary)
        onsets = np.where(d == 1)[0]
        offsets = np.where(d == -1)[0]

        # Match onsets with offsets
        if len(offsets) > len(onsets):
            if onsets[0] > offsets[0]:
                offsets = offsets[1:]
            else:
                offsets = offsets[:-1]
        elif len(offsets) < len(onsets):
            if onsets[0] > offsets[0]:
                onsets = onsets[1:]
            else:
                onsets = onsets[:-1]

        return onsets + 1, offsets

    # %%
    @staticmethod
    def pixels2degrees(pixels,viewing_dist = 0.63,screen_res = [1920,1080],
                       screen_size = [0.528, 0.297],dim = 'h',
                       center_coordinate_system = True):
        '''
        Converts from pixels to degrees
        Pixels can be a single value or a 1D array
        dim - horizontal 'h', or vertical 'v'
        '''

        # Conversion in horizontal or vertical dimension?
        if dim == 'h':
            res =  screen_res[0]
            size = screen_size[0]
        else:
            res = screen_res[1]
            size = screen_size[1]

        # SMIs coordinate-system starts in the upper left corner and psychopy's in
        # the center of the screen. y-axis is defined in different directions
        if center_coordinate_system:
            if dim == 'h':
                pixels = pixels - res/2.0
            else:
                pixels = (pixels*-1 + res/2.0)

        meter_per_pixel = size/res
        meters = pixels*meter_per_pixel

        alpha = 180.0/np.pi*(2*np.arctan(meters/viewing_dist/2.0))

        return alpha

    # %%
    def _merge_blinks(self, blink_onsets, blink_offsets, min_dur, min_separation,
                     additional_params=[]):
        """

        Merges blinks close together, and removes short blinks
        Args:
            blink_onsets (list): onsets of blinks (ms)
            blink_offsets (list): offsets of blinks (ms)
            min_dur (int): minimum duration of blink (ms)
            min_separation (int): minimal duration between blinks (ms, those with smaller are merged)

        Returns:
            blinks (list): list with onset, offset, duration

        """


        # Merge blink candidate close together, and remove short, isolated ones
        new_onsets = []
        new_offsets = []
        new_parameters = []
        change_onset = True

        for i, onset in enumerate(blink_onsets):
            # print(i, blink_onsets[i])
            if change_onset:
                temp_onset = blink_onsets[i]

            if i < len(blink_onsets) - 1:
                if ((blink_onsets[i+1] - blink_offsets[i])) < min_separation:

                    # if change_onset:
                    #     temp_onset = onsets[i]
                    change_onset = False
                else:
                    change_onset = True

                    # Remove blink with too short duration
                    if ((blink_offsets[i] - temp_onset)) < min_dur:
                        continue

                    new_onsets.append(temp_onset)
                    new_offsets.append(blink_offsets[i])
                    if len(additional_params) > 0:
                        new_parameters.append(additional_params[i, :])
            else:

                # # Remove blink with too short duration
                if ((blink_offsets[i] - temp_onset)) < min_dur:
                    continue

                new_onsets.append(temp_onset)
                new_offsets.append(blink_offsets[i])
                if len(additional_params) > 0:
                    new_parameters.append(additional_params[i, :])

        # Compute durations and convert to array
        blinks = []
        for i in range(len(new_onsets)):
            dur = new_offsets[i] - new_onsets[i]

            if len(additional_params) > 0:
                blinks.append([new_onsets[i], new_offsets[i], dur] +
                              list(new_parameters[i]))
            else:
                blinks.append([new_onsets[i], new_offsets[i], dur])

        # print(len(blink_onsets), len(new_onsets))

        return blinks

    # %%
    def blink_detector_pupil(self, t, pupil_signal, Fs, gap_dur=20, min_dur=20,
                             remove_outliers = False, min_separation=50):
        '''
        Args:
            t (np.array): DESCRIPTION.
            pupil_signal (np.array): DESCRIPTION.
            Fs (int): Sample rate.
            gap_dur (int, optional): Defaults to 20
            min_dur (int, optional):  Defaults to 20.
            min_separation (int, optional): blinks closer in time are merged. Defaults to 100.

        Returns:
            df (dataframe): .

        '''

        # Remove outliers?
        if remove_outliers:
            if np.isnan(self.settings.window_len):
                window_len_samples = len(pupil_signal)
            else:
                window_len_samples = int(Fs / 1000 * self.settings.window_len) # in ms window over which to exclude outliers

            ps = pupil_signal.copy()
            ps[ps < self.settings.min_pupil_size] = np.nan

            for k in np.arange(len(ps) - window_len_samples + 1):
                temp = pupil_signal[k : (k + window_len_samples)].copy()

                if len(temp) == 0:
                    continue

                m = np.nanmean(temp)
                sd = np.nanstd(temp)
                idx = (temp > (m + self.settings.treshold_SD * sd)) | (temp < (m - self.settings.treshold_SD * sd))
                temp[idx] = np.nan
                ps[k : (k + window_len_samples)] = temp

            pupil_signal = ps

        # Interpolate short periods of data loss
        pupil_signal = preprocessing.interpolate_nans(t, pupil_signal,
                                                              gap_dur=gap_dur)

        # Convert to bounds and clean up
        onsets, offsets = self._binary2bounds(np.isnan(pupil_signal) * 1)

        # Convert onsets/offsets to ms
        blinks = []
        for i, onset in enumerate(onsets):
            dur = t[offsets[i]] - t[onset]
            blinks.append([t[onset], t[offsets[i]], dur])

        # Remove blinks with on-, or offsets that happened in a period of missing data
        # (i.e., where samples are completely lost, for some reason)
        idx = np.where(np.diff(t) > (2 * 1/Fs * 1000)) # Missing data where deltaT > 2 * 1/Fs

        for i, blink in enumerate(blinks):
            for idx_k in idx[0]:
                if np.logical_and(blink[0] >= t[idx_k], blink[0] <= t[idx_k + 1]) or \
                   np.logical_and(blink[1] >= t[idx_k], blink[1] <= t[idx_k + 1]) :
                    blinks.pop(i)
                    break

        onsets = [b[0] for b in blinks]
        offsets = [b[1] for b in blinks]

        # Merge blinks closer than x ms, and remove short blinks
        blinks = self._merge_blinks(onsets, offsets, min_dur, min_separation)
        df = pd.DataFrame(blinks,
                          columns=['onset', 'offset', 'duration'])
        return df

    # %%
    def blink_detector_eo(self, t, eye_openness_signal, Fs, gap_dur=30,
                       filter_length=25,
                       width_of_blink=15,
                       min_separation=100,
                       plot_on=True):
        """

        Args:
            t - time in ms
            eye_openness_signal (1d numpy array): eye openness signal for left or right eye
            Fs (int): sampling frequency of the eo data
            gap_dur (int): interpolate gaps shorter than this duration (ms)
            filter_length (int): length of SG filter (ms)
            width_of_blink (int): min width of blink (ms)
            min_separation (int): min separation between blink peaks (ms)
            plot_on (boolean): plot the results?

        Returns:
            df - pandas dataframe with blink parameters

        """

        ms_to_sample = Fs / 1000
        sample_to_ms = 1000 / Fs


        # Assumes the eye is mostly open during the trial
        fully_open = np.nanmedian(eye_openness_signal, axis=0)
        min_amplitude = fully_open * self.settings.min_amplitude  # Equivalent to height in 'find_peaks'

        # detection parameters in samples
        distance_between_blinks = 1
        width_of_blink = width_of_blink * ms_to_sample
        filter_length = preprocessing.nearest_odd_integer(filter_length * ms_to_sample)

        # Interpolate gaps
        eye_openness_signal = preprocessing.interpolate_nans(t, eye_openness_signal,
                                                              gap_dur=int(gap_dur))

        # Filter eyelid signal and compute
        eye_openness_signal_filtered = savgol_filter(eye_openness_signal, filter_length, 2,
                                       mode='nearest')
        eye_openness_signal_vel = savgol_filter(eye_openness_signal, filter_length, 2,
                                           deriv=1,  mode='nearest') * Fs

        # Velocity threshold for on-, and offsets
        T_vel = stats.median_abs_deviation(eye_openness_signal_vel, nan_policy='omit') * 3

        # Turn blink signal into something that looks more like a saccade signal
        eye_openness_signal_inverse = (eye_openness_signal_filtered -
                                       np.nanmax(eye_openness_signal_filtered)) * -1
        peaks, properties = find_peaks(eye_openness_signal_inverse, height=None,
                                       distance=distance_between_blinks,
                                       width=width_of_blink)

        # Filter out not so 'prominent peaks'
        '''
        The prominence of a peak may be defined as the least drop in height
         necessary in order to get from the summit [peak] to any higher terrain.
        '''
        idx = properties['prominences'] > min_amplitude
        peaks = peaks[idx]
        for key in properties.keys():
            properties[key] = properties[key][idx]

        # Find peak opening/closing velocity by searching for max values
        # within a window from the peak
        blink_properties = []
        for i, peak_idx in enumerate(peaks):

            # Width of peak
            width = properties['widths'][i]

            ### Compute opening/closing velocity
            # First eye opening velocity (when eyelid opens after a blink)
            peak_right_idx = np.nanargmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])
            peak_right_idx = np.nanmin([peak_right_idx, len(eye_openness_signal_vel)])
            idx_max_opening_vel = int(peak_idx + peak_right_idx)
            time_max_opening_vel = t[idx_max_opening_vel]
            opening_velocity = np.nanmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])

            # Then eye closing velocity (when eyelid closes in the beginning of a blink)
            peak_left_idx = width - np.nanargmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx]) + 1
            peak_left_idx = np.nanmax([peak_left_idx, 0])
            idx_max_closing_vel = int(peak_idx - peak_left_idx + 1)
            time_max_closing_vel = t[idx_max_closing_vel]
            closing_velocity = np.nanmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx])

            # Identify on and offsets (go from peak velocity backward/forward)
            temp = eye_openness_signal_vel[idx_max_opening_vel:]
            if np.any(temp <= (T_vel / 3)):
                offset = np.where(temp <= (T_vel / 3))[0][0]
            else:
                offset = len(temp)

            # make sure the blink period stop when encountering nan-data
            # If it does, make the opening phase parameters invalid
            if np.any(np.isnan(temp)):
                offset_nan = np.where(np.isnan(temp))[0][0]
                offset = np.min([offset, offset_nan])

            offset_idx = int(idx_max_opening_vel + offset - 1)

            temp = np.flip(eye_openness_signal_vel[:idx_max_closing_vel])
            if np.any(temp >= -T_vel):
                onset = np.where(temp >= -T_vel)[0][0]
            else:
                onset = 0

            if np.any(np.isnan(temp)):
                onset_nan = np.where(np.isnan(temp))[0][0]
                onset = np.min([onset, onset_nan])

            onset_idx = int(idx_max_closing_vel - onset)


            # Compute openness at onset, peak, and offset
            openness_at_onset = eye_openness_signal_filtered[onset_idx]
            openness_at_offset = eye_openness_signal_filtered[offset_idx]
            openness_at_peak = eye_openness_signal_filtered[peak_idx]

            # Compute amplitudes for closing and opening phases
            closing_amplitude = np.abs(openness_at_onset - openness_at_peak)
            opening_amplitude = np.abs(openness_at_offset - openness_at_peak)

            distance_onset_peak_vel = np.abs(eye_openness_signal_filtered[onset_idx] -
                                             eye_openness_signal_filtered[idx_max_closing_vel]) # mm
            timediff_onset_peak_vel = np.abs(onset_idx - idx_max_closing_vel) * sample_to_ms # ms

            # Onset and peak cannot be too close in space and time
            if (distance_onset_peak_vel < 0.1) or (timediff_onset_peak_vel < 10):
                if self.settings.debug:
                    print('Peak to close to onset')
                continue

            if np.min([opening_velocity, np.abs(closing_velocity)]) < (T_vel * 2):
                if self.settings.debug:
                    print('Blink velocity too low')
                continue

            blink_properties.append([t[onset_idx],
                                     t[offset_idx],
                                     t[offset_idx] - t[onset_idx],
                                     t[peak_idx],
                                     openness_at_onset, openness_at_offset,
                                     openness_at_peak,
                                     time_max_opening_vel,
                                     time_max_closing_vel,
                                     opening_velocity, closing_velocity,
                                     opening_amplitude, closing_amplitude])

        # Merge blinks too close together in time
        blink_temp = np.array(blink_properties)
        blink_onsets = blink_temp[:, 0]
        blink_offsets = blink_temp[:, 1]

        bp =  self._merge_blinks(blink_onsets, blink_offsets, width_of_blink, min_separation,
                         additional_params=blink_temp[:, 3:])

        # Convert to dataframe
        df = pd.DataFrame(bp,
                          columns=['onset', 'offset', 'duration',
                                   'time_peak',
                                   'openness_at_onset',
                                   'openness_at_offset',
                                   'openness_at_peak',
                                   'time_peak_opening_velocity',
                                   'time_peak_closing_velocity',
                                   'peak_opening_velocity',
                                   'peak_closing_velocity',
                                   'opening_amplitude',
                                   'closing_amplitude'])

        df.openness_at_peak[df.openness_at_peak < 0] = 0

        return df, eye_openness_signal_vel

    # %%
    def plot_blink_detection_results(self,
                                     t,
                                     eye_openness_signal,
                                     eye_openness_signal_vel,
                                     df_blink,
                                     pid_name,
                                     trial_name,
                                     eye,
                                     pupil_signal=[],
                                     df_blink_pupil = pd.DataFrame(),
                                     xy = []):

        if len(xy) > 0:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16*2,9*2))
            ax[2].plot(t, BlinkDetector.pixels2degrees(xy[:, 0] * 1920))
            ax[2].plot(t, BlinkDetector.pixels2degrees(xy[:, 1] * 1080, dim='v'))
            ax[2].set_xlabel('Time (ms)')
            ax[2].set_ylabel('Gaze position (deg)')
            ax[2].legend(['x', 'y'])
        else:
            fig, ax = plt.subplots(2, 1, sharex=True)


        ax[0].plot(t, eye_openness_signal)

        peaks = np.array(df_blink.time_peak)
        peak_idx = np.searchsorted(t, peaks)
        ax[0].plot(peaks, eye_openness_signal[peak_idx.astype(int)], "kx", ms=20, lw = 5)
        ax[0].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Eye openness (mm)')
        ax[0].set_title(f'{pid_name}_{trial_name},  {eye},  {len(df_blink)}, {len(df_blink_pupil)}, \
     {df_blink.duration.mean():.2f}, {df_blink_pupil.duration.mean():.2f}')

        if len(pupil_signal) > 0:
            ax[0].plot(t, pupil_signal * 2, 'k')


        ax[1].plot(t, eye_openness_signal_vel)
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Eye openness velocity (mm/s)')

        # Go through the dataframe and plot one blink at the time
        for i, row in df_blink.iterrows():

            if row.openness_at_peak <= self.settings.full_blink_max_opening:
                ax[0].axvspan(row.onset, row.offset, alpha=0.5, color='red')
            else:
                ax[0].axvspan(row.onset, row.offset, alpha=0.5, color='green')

            ax[1].plot(row.time_peak_closing_velocity, row.peak_closing_velocity, 'ro')
            ax[1].plot(row.time_peak_opening_velocity, row.peak_opening_velocity, 'go')

            idx = np.searchsorted(t, row.offset)
            ax[0].plot(row.offset, eye_openness_signal[idx], 'rv', ms=10)
            ax[1].plot(row.offset, eye_openness_signal_vel[idx], 'rv', ms=10)
            idx = np.searchsorted(t, row.onset)
            ax[0].plot(row.onset, eye_openness_signal[idx], 'gv', ms=10)
            ax[1].plot(row.onset, eye_openness_signal_vel[idx], 'gv', ms=10)

            if row.openness_at_peak <= self.settings.full_blink_max_opening:
                ax[1].axvspan(row.onset, row.offset, alpha=0.5, color='red')
            else:
                ax[1].axvspan(row.onset, row.offset, alpha=0.5, color='green')

            # Go through the dataframe and plot one blink at the time
            for i, row in df_blink_pupil.iterrows():
                ax[0].plot([row.onset, row.offset],
                           [6, 6],
                           alpha=0.5, color='c', ms=10, lw=5)

        plt.show()

        if self.settings.plot_on and self.settings.save_fig:
            plt.savefig(pid_name + '_' + trial_name + '.pdf')
