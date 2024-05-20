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
    Ts: float = 1 / fs            # the sampling period of the sound card

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
    
    # adjust the code to mimic the capturing of the AGC
    def generate_rgpo_sig(self, delay_factor=4):
        '''generate a signal that gradually pulls from the range and then stops'''
        init_delay = self.pw_samples + int(self.pw_samples / delay_factor)
        recurring_delay = int(self.pw_samples / (4 * delay_factor))
        append_pulse = True
        track_range_gate = init_delay + recurring_delay
        track_pulse_count = 0

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
        delay_pulses = 5
        print(f'Delaying the range gate pull-off signal by {delay_pulses} pulses')
        rgpo_sig = np.append(rgpo_sig, np.zeros(delay_pulses * len(self.pri_zeros)))

        # capture the AGC
        capture_pulses = 9
        print(f'Capturing the AGC with {capture_pulses} pulses')
        for ii in range(delay_pulses, capture_pulses): 
            rgpo_sig = np.append(rgpo_sig, pulse)
            rgpo_sig = np.append(rgpo_sig, self.lstn_zeros)

        j = 0
        # append the pulse signal to rgpo_sig while gradually pulling off from the range gate
        while append_pulse or (track_pulse_count <= (self.num_pulses - delay_pulses - capture_pulses)):
            rgpo_sig = np.append(rgpo_sig, pulse)
            rgpo_sig = np.append(rgpo_sig, self.lstn_zeros)
            rgpo_sig = np.append(rgpo_sig, recur_delay_zeros)
            j += 1

            # update track_range_gate variable and verify its within the gate
            #print(track_range_gate)
            track_range_gate += recurring_delay
            append_pulse = True if track_range_gate < self.total_samples else False
            track_pulse_count += 1

        print(f'took {j} to walk away from the range gate')
        # add the zeros to match expected size
        rem_samples = int(self.fs * self.duration) - len(rgpo_sig)
        rem_zeros = np.zeros(rem_samples)
        rgpo_sig = np.append(rgpo_sig, rem_zeros)
        self.plotter(rgpo_sig, 1)
        rgpo_sig = np.array(rgpo_sig).astype(np.float32)

        return rgpo_sig

    def coverPulse(self, cover_fc):
        '''generates a cover pulse signal to prevent operation of the sonar'''
        min_distance = self.R_min + 0.5
        cover_sig = []

        # shorten the delay and listening times to cater for a longer cover pulse
        adj_pulse_width = (2 * min_distance) / self.sound_speed
        adj_samples = ceil(adj_pulse_width / ( 1 / self.fs)) - self.pw_samples
        adj_samples = int(adj_samples / 2)
        adj_delay_samples = self.delay_samples[:len(self.delay_samples) - adj_samples]
        adj_lstn_samples = self.lstn_zeros[:self.lstn_samples - (2 * adj_samples)]
        adj_pw_samples = self.pw_samples + (2 * adj_samples)

        print(f"The cover pulse signal has {adj_pw_samples} pulse width samples and {len(adj_lstn_samples)} dead time samples." )       

        # generate an LFM pulse with these parameters
        adj_pulse = chirp(
            np.linspace(0, adj_pulse_width, adj_pw_samples), 
            f0=cover_fc,
            f1=cover_fc+self.bwidth, 
            t1=adj_pulse_width, 
            method='linear'
        ).astype(np.float32)

        # generate the cover signal
        samps = int((self.total_samples - adj_pw_samples) / 2)
        cover_sig = np.append(cover_sig, self.delay_samples)
        cover_sig = np.append(cover_sig, np.zeros(864))

        # delay generation of cover pulses to mimic DRFM reception, storage, and retransmission
        delay_pulses = 8
        print(f'The cover pulse to be generated after the {delay_pulses}th LFM pulse')
        cover_sig = np.append(cover_sig, np.zeros(delay_pulses * len(self.pri_zeros)))

        # generate cover pulse signals for the remaining pulses
        for i in range(delay_pulses, self.num_pulses):
            cover_sig = np.append(cover_sig, adj_pulse)
            cover_sig = np.append(cover_sig, np.zeros(int(2 * samps)))

        #cover_sig = np.append(cover_sig, adj_delay_samples)
        cover_sig = np.append(cover_sig, np.zeros(int(self.duration * self.fs - len(cover_sig))))
        cover_sig = np.array(cover_sig).astype(np.float32)

        return cover_sig
    
    def velocity_deception(self, vgpo_fc):
        '''Generates a velocity gate pull-off signal'''
        vgpo_sig = []
        fshift = 0.25        # Hz
        fshift_range = 2.3  # the range at which fshift is placed + minimum
        counter = 2         # a multiplier for the frequency increment

        # generate a pulse signal shifted in frequency
        fshift_pulse = chirp(
            np.linspace(0, self.pulse_width, self.pw_samples), 
            f0=vgpo_fc,
            f1=vgpo_fc+self.bwidth, 
            t1=self.pulse_width, 
            method='linear'
        ).astype(np.float32)

        # delay transmission to mimic DRFM signal capture, storage, modification and retransmission
        delay_trans = rdom.randint(3,6)
        print(f'The vgpo signal is delayed up until the {delay_trans}th pulse.')
        vgpo_sig = np.append(vgpo_sig, self.delay_samples)
        for i in range(delay_trans):
            vgpo_sig = np.append(vgpo_sig, self.pri_zeros)

        # mimic capturing of the AGC at a certain range
        fshift_samples = self.pw_samples + ceil(((2 * fshift_range) / self.sound_speed) / (1 / self.fs ))
        fshift_zeros = np.zeros(fshift_samples)
        vgpo_sig = np.append(vgpo_sig, fshift_zeros)
        capturing_agc = rdom.randint(3, 5)
        print(f'Capturing the AGC with {capturing_agc} pulses of the same frequency')
        for i in range(capturing_agc):
            vgpo_sig = np.append(vgpo_sig, fshift_pulse)
            vgpo_sig = np.append(vgpo_sig, self.lstn_zeros)

        # calculate the maximum unambiguous velocity of the system
        v_max = self.sound_speed**2 / (8 * self.fc * self.R_max)
        v_shift = (self.sound_speed * fshift) / (2 * vgpo_fc)
        v_update = v_shift

        # increase the frequency shift to mimic a moving away target
        while v_update < v_max:
            # generate the pulse signal with a different frequency
            fshift_pulse = chirp(
                np.linspace(0, self.pulse_width, self.pw_samples), 
                f0=vgpo_fc+(counter*fshift),
                f1=vgpo_fc+self.bwidth+(counter*fshift), 
                t1=self.pulse_width, 
                method='linear'
            ).astype(np.float32)
            
            # append the doppler shifted pulse to the signal
            vgpo_sig = np.append(vgpo_sig, fshift_pulse)
            vgpo_sig = np.append(vgpo_sig, self.lstn_zeros)

            # update the variables for a frequency increment
            counter += 1
            v_update += v_shift
        
        # append all the remaining zeros
        rem_zeros = np.zeros(int((self.duration * self.fs) - len(vgpo_sig)))
        vgpo_sig = np.append(vgpo_sig, rem_zeros)
        vgpo_sig = np.array(vgpo_sig).astype(np.float32)

        return vgpo_sig
    
    def mft(self):
        '''Generates a PRI with multiple false targets'''
        # define the pulse characteristics you want in your false target
        # these have to be within the PRI of your pulse
        mft_sig = []
        pri_mft_sig = []
        track_samples = 0

        # adding the delay before the first false target is added 
        pulse_delay = 4
        pulse_zeros = np.zeros(int(pulse_delay * len(self.pri_zeros)))
        mft_sig = np.append(mft_sig, self.delay_samples)
        mft_sig = np.append(mft_sig, pulse_zeros)

        # adding the delay before introducing a false target pulse
        sample_delay = int((self.lstn_samples - self.pw_samples) / 2)
        sample_zeros = np.zeros(sample_delay)
        pri_mft_sig = np.append(pri_mft_sig, sample_zeros)

        # defining 1st target (stationery)
        pulse = chirp(
            np.linspace(0, self.pulse_width, self.pw_samples), 
            f0=self.fc,
            f1=self.fc+self.bwidth, 
            t1=self.pulse_width, 
            method='linear'
        ).astype(np.float32)
        pri_mft_sig = np.append(pri_mft_sig, np.zeros(self.pw_samples)) # should be pulse
        
        another_delay_samples = int(sample_delay / 2)
        ads_zeros = np.zeros(another_delay_samples)
        pri_mft_sig = np.append(pri_mft_sig, ads_zeros)
        track_samples = sample_delay + self.pw_samples + another_delay_samples

        # adding the second false target pulse (moving)
        '''pulse = chirp(
            np.linspace(0, self.pulse_width, self.pw_samples), 
            f0=self.fc+20,
            f1=self.fc+self.bwidth+20, 
            t1=self.pulse_width, 
            method='linear'
        ).astype(np.float32)'''
        t = np.linspace(0, self.pulse_width, self.pw_samples)
        chirp_slope = (12e3-9e3)/self.pulse_width
        inst_freq = 10.5e3 + (12e3-9e3)/2*(2*t/self.pulse_width - 1)
        phase = 2 * np.pi * np.cumsum(inst_freq * t)

        doppler_shift = (2 * 0.05)*  10.5e3
        shifted_phase = np.angle(self.pulse) + np.pi/2.5

        pulse = np.cos(shifted_phase)
        pri_mft_sig = np.append(pri_mft_sig, pulse)
        pri_mft_sig = np.append(pri_mft_sig, ads_zeros)
        track_samples += self.pw_samples + another_delay_samples

        # the number of samples at this point should equal a PRI
        if track_samples != self.total_samples:
            print(f'The number of pri samples and mft pri samples not equal. track_samples = {track_samples}')
            exit(1)

        # when you are done with your desired false targets, add them to your stream of pulses
        # make sure you add some delay of pulses before adding them
        for ii in range(pulse_delay, self.num_pulses):
            mft_sig = np.append(mft_sig, pri_mft_sig)

        mft_sig = np.append(mft_sig, self.delay_samples)
        mft_sig = np.array(mft_sig).astype(np.float32)

        return mft_sig
    
    def noise_effect(self, f0, noise_bwidth, order=6):
        '''generates a signal with varying bandwidth''' 
        noise_sig = []
        noise_sig = np.append(noise_sig, self.delay_samples)

        # generate the white noise signal
        noise = np.random.randn(self.num_pulses * self.total_samples)

        # design a bandpass filter
        from scipy.signal import lfilter, butter
        nyquist_rate = self.fs / 2
        lowcut = (f0 - noise_bwidth / 2) / nyquist_rate 
        highcut = (f0 + noise_bwidth / 2) / nyquist_rate
        print(f'The lower and upper cutoff frequencies are {lowcut} and {highcut}')
        b, a = butter(order, [lowcut, highcut], btype='bandpass')

        # apply the filter
        filtered_noise = lfilter(b, a, noise)

        plt.subplot(111)
        F_X = np.fft.fftshift(np.fft.fft(filtered_noise, 1024))
        freq = np.arange(-self.fs / 2 , self.fs / 2, self.fs / len(F_X))
        plt.plot(freq / 1000, np.abs(F_X))
        plt.xlabel("Frequency (kHz)")
        plt.ylabel('Magnitude')
        plt.xticks(np.arange(-self.fs / (2 * 1000), self.fs / (2 * 1000), 2))
        plt.grid(True)
        plt.show()

        # append to the noise signal 
        noise_sig = np.append(noise_sig, filtered_noise)
        noise_sig = np.append(noise_sig, self.delay_samples)
        noise_sig = np.array(noise_sig).astype(np.float32)

        return noise_sig

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

