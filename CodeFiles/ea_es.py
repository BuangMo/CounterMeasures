from math import ceil
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random as rdom

@dataclass
class Initialiser:
    eattack_select: int          # electronic attack option
    fc: float                     # the centre frequency of the transmitted waveform
    bwidth: float = 3e3           # the bandwidth of the radar
    fs: float = 48e3              # the sampling frequency of the sound card
    sound_speed: float = 343      # the speed of sound (343 m/s)
    R_max: float = 7              # the maximum distance (in m) that the system must be capable of
    R_min: float = 3              # the minimum distance at which system should still detect targets
    num_pulses: int = 32          # the number of pulses for the transmitted waveform
    volume: float = 1             # the volume of the sound output
    duration: int = 2             # the duration to play the signal

class TXSignal(Initialiser):
    def generateTXSignal(self, operation):
        '''generates the lFM signal'''
        tx_signal = []

        # calculate signal properties
        self.pulse_width = (2 * self.R_min) / self.sound_speed
        PRI = (2 * self.R_max) / self.sound_speed

        self.lstn_samples = ceil((PRI - self.pulse_width) / (1 / self.fs ))
        self.pw_samples = ceil(self.pulse_width / (1 / self.fs ))
        self.total_samples = self.lstn_samples + self.pw_samples

        print(f"The pulse signal has {self.pw_samples} pulse width samples and {self.lstn_samples} dead time samples." )       

        # construction of the signals
        self.lstn_zeros = np.zeros(self.lstn_samples)
        self.delay_samples = np.zeros(int((self.duration * self.fs - self.num_pulses * (self.pw_samples + self.lstn_samples)) / 2))
        self.pri_zeros = np.zeros(self.total_samples)

        # message signal
        tx_signal = np.append(tx_signal, self.delay_samples)

        # carrier signal and template
        if operation == 0:          # constructs a frequency constant waveform
            self.pulse = chirp(np.linspace(0, self.pulse_width, self.pw_samples), f0=self.fc,
                        f1=self.fc+self.bwidth, t1=self.pulse_width, method='linear').astype(np.float32)
            
            for i in range(self.num_pulses):
                tx_signal = np.append(tx_signal, self.pulse)
                tx_signal = np.append(tx_signal, self.lstn_zeros)
        else:                       # constructs a frequency agile waveform
            self.pulse = self.frequency_agility(self.pulse_width, self.fc)
        
            for i in range(self.num_pulses):
                tx_signal = np.append(tx_signal, self.pulse[i,:])
                tx_signal = np.append(tx_signal, self.lstn_zeros)

        tx_signal = np.append(tx_signal, self.delay_samples)
        tx_signal = np.array(tx_signal).astype(np.float32)

        return tx_signal

    def frequency_agility(self, pulse_width, start_fc=8e3):
        """ Generates generates frequency agile pulse templates"""
        templates = np.zeros((self.num_pulses, self.pw_samples), dtype=np.float32)
        f_init = 0
        factor = 0
        
        for i in range(self.num_pulses):           
            f_init = factor * self.bwidth + start_fc
            templates[i,:] = chirp(np.linspace(0, pulse_width, self.pw_samples), f0=f_init,
                    f1=f_init+self.bwidth, t1=pulse_width, method='linear').astype(np.float32)

            if (i+1)%4 == 0:
                factor = 0
            else:
                factor += 1

        return templates
    
    def generate_rgpo_sig(self, delay_factor=4):
        '''generate a signal that gradually pulls from the range and then stops'''
        init_delay = self.pw_samples + int(self.pw_samples / delay_factor)
        recurring_delay = int(self.pw_samples / (4 * delay_factor))
        append_pulse = True
        track_range_gate = init_delay + recurring_delay

        # create zeros samples for initial delay and recurring delay
        init_delay_zeros = np.zeros(init_delay)
        recur_delay_zeros = np.zeros(recurring_delay)

        # create the pulse signal
        pulse = chirp(
            np.linspace(0, self.pulse_width, self.pw_samples), 
            f0=self.fc,
            f1=self.fc+self.bwidth, 
            t1=self.pulse_width, 
            method='linear'
        ).astype(np.float32)

        # create a range gate pull-off signal with initial delay
        rgpo_sig = []
        rgpo_sig = np.append(rgpo_sig, self.delay_samples)
        rgpo_sig = np.append(rgpo_sig, init_delay_zeros)

        # delay the rgpo by the number of pulses to make it realistic
        delay_pulses = rdom.randint(4, 10)
        print(f'Delaying the range gate pull-off signal by {delay_pulses} pulses')
        for i in range(delay_pulses):
            rgpo_sig = np.append(rgpo_sig, self.pri_zeros)

        # append the pulse signal to rgpo_sig while gradually pulling off from the range gate
        while append_pulse:
            rgpo_sig = np.append(rgpo_sig, pulse)
            rgpo_sig = np.append(rgpo_sig, self.lstn_zeros)
            rgpo_sig = np.append(rgpo_sig, recur_delay_zeros)

            # update track_range_gate variable and verify its within the gate
            track_range_gate += recurring_delay
            append_pulse = True if track_range_gate < self.total_samples else False

        # add the zeros to match expected size
        rem_samples = int(self.fs * self.duration) - len(rgpo_sig)
        rem_zeros = np.zeros(rem_samples)
        rgpo_sig = np.append(rgpo_sig, rem_zeros)
        rgpo_sig = np.array(rgpo_sig).astype(np.float32)

        return rgpo_sig

    def plotter(self, f_x, plot_sel=2):
        '''Plots the time series and frequency spectrums of the given signal'''
        # performs the DFT of the given signal
        F_X = np.fft.fftshift(np.fft.fft(f_x, 1024))

        # defines the time and freq range of the signal
        freq = np.arange(-self.fs / 2, self.fs / 2, self.fs / len(F_X))
        t = np.arange(len(f_x)) / self.fs

        # plots the time and frequency spectrum
        if plot_sel == 1:       # plots the input function in the time domain as a function of time
            plt.subplot(111)
            plt.plot(t, f_x)
            plt.xlabel("time (s)")
            plt.ylabel('Magnitude')
            plt.grid(True)
        elif plot_sel == 2:     # plots the input function in the time domain as a function of sample
            plt.subplot(111)
            plt.plot(np.arange(f_x.size) / 1000, f_x)
            plt.xlabel("Samples (kHz)")
            plt.ylabel('Magnitude')
            plt.grid(True)
        elif plot_sel == 3:     # plots the input function as a function of frequency
            plt.subplot(111)
            plt.plot(freq / 1000, np.abs(F_X))
            plt.xlabel("Samples (kHz)")
            plt.ylabel('Magnitude')
            plt.grid(True)
        else:
            _, ax = plt.subplots(1, 2)
            ax[0].plot(t, f_x)
            ax[0].set(xlabel="time (s)", ylabel='Magnitude')
            ax[0].set_title('(a)', loc='left')
            ax[0].grid(True)

            ax[1].plot(freq / 1000, np.abs(F_X))
            ax[1].set(xlabel="Frequency (kHz)", ylabel='Magnitude')
            ax[1].set_title('(b)', loc='left')
            ax[1].grid(True)

        plt.show()

