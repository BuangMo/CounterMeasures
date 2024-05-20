from ea_es import TXSignal
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
import pyaudio, queue, time
from threading import Thread
from scipy.signal import hilbert, find_peaks

class ELB780_2:
    def __init__(self, eattack_select, fc, operation):
        '''Initialises the project'''
        self.operation = operation                  # mode of sonar operation
        self.eattack_select = eattack_select        # option for the electronic attack signal
        self.chunk = 1024                           # number of samples played or captured
        self.rcv = []                               # recorded echo data
        self.tx_signal = TXSignal(eattack_select, fc)

        # pyAudio params
        self.p = pyaudio.PyAudio()                  # Create an interface to PortAudio
        self.format = pyaudio.paFloat32
        self.channel = 1 if self.tx_signal.eattack_select == 0 else 2

        plt.close('all')

    def emitter(self):
        '''Captures the echoes of the transmitted waveform'''
        try:
            ostream = self.p.open(
                format=self.format,
                channels=self.channel,
                rate=int(self.tx_signal.fs),
                output=True
            )

            start_time = time.time()
            print( "Playing..." )
            ostream.write(self.output_bytes)
            print("Played sound for {:.2f} seconds".format(time.time() - start_time))

            # Stop and Close the output stream
            ostream.stop_stream()
            ostream.close()
        except (OSError, NameError, ValueError) as error:
            self.p.terminate()
            print(f'A {type(error).__name__} has occured')

    def listener(self):
        '''Plays the output waveform through the speakers'''
        frames = []
        Qin = queue.Queue()
        
        try:
            istream = self.p.open(
                format=self.format,
                channels=1,
                rate=int(self.tx_signal.fs),
                input=True
            )

            print("Recording...")

            # captures the data
            for dt in range(0, int((self.tx_signal.duration * self.tx_signal.fs) / (self.chunk))):
                data = istream.read(self.chunk)
                frames = np.frombuffer(data, dtype=np.float32)
                Qin.put(frames)

            # Stop and Close the input stream
            istream.stop_stream()
            istream.close()

            # capture the recorded data in a list
            while(not Qin.empty()):
                rcv_data = Qin.get()
                self.rcv = np.append(self.rcv, rcv_data)
        except (OSError, NameError, ValueError) as error:
            self.p.terminate()
            print(f'A {type(error).__name__} has occured')

    def terminate_pa(self):
        '''Terminates the pyaudio binding to portaudio'''
        self.p.terminate()

    def pulse_compression(self):
        '''Applies the matched filter to the received data'''
        print("Applying the matched filter")
        #self.tx_signal.plotter(self.rcv, 1)
        self.mfiltered = np.convolve(
            hilbert(self.rcv), 
            np.conj(np.flipud(hilbert(self.tx_signal.pulse))), 
            mode='same'
        )
        self.tx_signal.plotter(self.mfiltered.real, 2)

    def find_1st_peak(self):
        stop_search = int(self.mfiltered.size / 2)
        peaks, _ = find_peaks(np.abs(self.mfiltered[:stop_search]), height=4)
        trace_peak = 0

        # run through the peaks to find the difference in samples between consecutive peaks
        for i in range(peaks.size):
            distance = peaks[i + 1] - peaks[i]
            if distance > 20:
                trace_peak = i
                break

        if peaks[trace_peak] < 22e3:
            print(peaks[trace_peak])
            exit(444)
        return peaks[trace_peak]

    def data_matrix(self):
        print(f"Arrange the data into a matrix of {self.tx_signal.num_pulses} pulses.")
        
        # get the start of the first fast time sampling point
        origin_smpl = self.find_1st_peak()
        print(f'calculated max at {origin_smpl}')

        #gets the data into a matrix of complex numbers
        sampled = self.tx_signal.lstn_samples+self.tx_signal.pw_samples
        self.pri_data = np.zeros((sampled, self.tx_signal.num_pulses), dtype='complex_')
        starts, stops = 0, 0
        
        for i in range(self.tx_signal.num_pulses):
            starts = origin_smpl+i*sampled
            stops = origin_smpl+(i+1)*sampled
            self.pri_data[:,i] = self.mfiltered[starts:stops]
            #print(f"{i}. Start={starts} stop={stops} origin={origin_smpl} difference={stops-starts}.")

        #integrate the pulses
        dt = np.mean(self.pri_data.T, axis=0)
        analytic_sigg = np.abs(dt)
        self.tx_signal.plotter(analytic_sigg, 1)

        t = np.arange(len(self.pri_data.T[0])) / self.tx_signal.fs
        i = 1
        path = 'C:/Users/Buang/Documents/python_projects/tukkies/00 ELB780/2024/02 Assignment2/Results/03 Cover Pulse/Averaged/'
        self.file_handling(path)
        print('Writing the figures to the folder...')
        for ii in self.pri_data.T:
            lable = 'pulse ' + str(i)
            i += 1
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(t, np.abs(ii), label=lable)
            ax.legend()
            ax.set(xlabel='time (s)', ylabel='Magnitude')
            ax.grid(True)
            fig.savefig(path + lable + '.png')
            #plt.show()
            plt.close(fig)

    def file_handling(self, path_to_folder):
        import pathlib
        import shutil

        try:
            # delete the folder
            folder_path = pathlib.Path(path_to_folder)
            shutil.rmtree(path_to_folder, ignore_errors=True)
            print('Folder deleted')
            # create a folder
            folder_path.mkdir()
        except FileNotFoundError:
            print("Folder doesn't exist")
            folder_path.mkdir()

    def signal_construction(self):
        '''Constructs the waveform to survey the environment'''
        esupport_sig = self.tx_signal.generateTXSignal(self.operation)
        smps = np.arange(esupport_sig.size) / 1000
        plt.subplot(111)

        # construct the bytes signal based on the value of the eattack_select variable
        if self.eattack_select == 1:         # range gate
            eattack_sig = self.tx_signal.generate_rgpo_sig()
            #plt.plot(smps, eattack_sig, label='rgpo signal')
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        elif self.eattack_select == 2:       # velocity gate pull-off
            eattack_sig = self.tx_signal.velocity_deception(3e3)
            #plt.plot(smps, eattack_sig)
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        elif self.eattack_select == 3:       # cover pulse
            eattack_sig = self.tx_signal.coverPulse(9e3)
            #self.tx_signal.plotter(eattack_sig, 5)
            plt.plot(smps, eattack_sig, label='cover pulse')
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        elif self.eattack_select == 4:       # multiple false targets
            eattack_sig = self.tx_signal.mft()
            #self.tx_signal.plotter(eattack_sig, 5)
            #plt.plot(smps, eattack_sig, label='mft signal')
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        elif self.eattack_select == 5:       # noise with bandwidth effects
            eattack_sig = self.tx_signal.noise_effect(10.5e3, 3e3)
            #self.tx_signal.plotter(eattack_sig, 5)
            #plt.plot(smps, eattack_sig, label='noise signal')
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        else:
            self.output_bytes = (self.tx_signal.volume * esupport_sig).tobytes()

        plt.plot(smps, esupport_sig, label='sonar signal')
        plt.legend()
        plt.xlabel('samples (k)')
        plt.ylabel('amplitude')
        plt.grid()
        plt.show()

    def doppler_processing(self):
        '''Applying the Doppler filter to the data matrix'''
        print( "Applying the Doppler filter to the data matrix" )

        # calculate range and Doppler resolutions
        delta_r = self.tx_signal.sound_speed / (2 * self.tx_signal.bwidth)
        delta_v = self.tx_signal.sound_speed / (2 * ((self.tx_signal.bwidth / 2) + self.tx_signal.fc) * (self.tx_signal.pw_samples * self.tx_signal.Ts))

        # calculate the number of range and Doppler Cells
        Nr = int(ceil(2 * self.tx_signal.R_max / delta_r))
        Nv = int(ceil(2 * self.tx_signal.fs / delta_v))

        # generate range and Doppler FFT grids
        r_grid = np.linspace(0, Nr*delta_r, Nr, endpoint=False)
        v_grid = np.linspace(-Nv/2, Nv/2-1, Nv)*delta_v
        num_bins = 128
        print(self.pri_data.shape)

        #generate range-Doppler map
        self.doppler_spectrum = np.zeros((self.tx_signal.total_samples, num_bins), dtype='complex_')
        for i in range(self.tx_signal.total_samples):
            self.doppler_spectrum[i,:] = np.fft.fftshift(np.fft.fft(self.pri_data[i,:], num_bins))

        # generate velocity grid
        lambda_c = self.tx_signal.sound_speed / ((self.tx_signal.bwidth / 2) + self.tx_signal.fc)
        v_max = lambda_c / (4 * ((self.tx_signal.pw_samples + self.tx_signal.lstn_samples) * self.tx_signal.Ts))
        v_grid = np.linspace(-v_max, v_max, Nv)

        # normalise the results of the range doppler plot
        max_value = np.max(np.abs(self.doppler_spectrum))
        self.doppler_spectrum1 = np.abs(self.doppler_spectrum) / max_value

        doppler = np.log10(np.abs(self.doppler_spectrum.T))

        self.extent = [r_grid[0], r_grid[-1]/2, -v_grid[-1], -v_grid[0]]
        plt.imshow(np.abs(self.doppler_spectrum1.T), aspect='auto', cmap='jet', extent=self.extent)
        plt.xlabel('Range (m)')
        plt.xticks(np.arange(0, 8, 1))
        plt.ylabel('Velocity (m/s)')
        #plt.title('Range-Doppler map')
        plt.colorbar(label='Intensity Level')
        plt.show()

    def cfar2Ddetection(self, guard_band, training_cells, threshold_factor):
        ''' applies the 2D CFAR filter to the data'''
        doppler_spectrum = self.doppler_spectrum1
        print(self.doppler_spectrum.shape)
        num_rows, num_cols = doppler_spectrum.shape
        threshold_map = np.zeros_like(doppler_spectrum)
        away_edges = 2 * (training_cells + guard_band) + 1

        for i in range(num_rows):
            for j in range(num_cols):
                # define the region of interest
                start_row_a = max(0, i - guard_band - training_cells)
                end_row_a = min(num_rows, i + guard_band + training_cells + 1)
                start_col_a = max(0, j - guard_band - training_cells)
                end_col_a = min(num_cols, j + guard_band + training_cells + 1)

                # define the region of delete
                start_row_del = max(0, i - guard_band)
                end_row_del = min(num_rows, i + guard_band + 1)
                start_col_del = max(0, j - guard_band)
                end_col_del = min(num_cols, j + guard_band + 1)

                # when away from the left edge
                if end_col_a > away_edges:
                    start_col_del = training_cells
                    end_col_del = 2 * training_cells + guard_band + 1

                # when away from the top row
                if end_row_a > away_edges:
                    start_row_del = training_cells
                    end_row_del = 2 * training_cells + guard_band + 1

                # caculation of the local mean excluding the celss under test (CUT)
                local_data_a = doppler_spectrum[start_row_a:end_row_a, start_col_a:end_col_a]
                mask = np.ones_like(local_data_a, dtype=bool)
                mask[start_row_del:end_row_del, start_col_del:end_col_del] = False
                local_data = local_data_a[mask]

                local_mean = np.mean(local_data)

                # calculation of the local standard deviation
                local_std = np.std(local_data)

                # calculate the threshold
                threshold = local_mean * threshold_factor #* local_std

                # if the cell value is greater than the threshold, mark it as 1, else 0
                if doppler_spectrum[i, j] < threshold:
                    threshold_map[i, j] = 100
                else:
                    threshold_map[i, j] = 0

        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.imshow(doppler_spectrum.T, cmap='jet', aspect='auto', extent=self.extent)
        plt.title("(a)", loc='left')
        plt.ylabel('Velocity (m/s)')
        plt.colorbar(label='Intensity Level')
        plt.xticks(np.arange(0, 7, 1))

        plt.subplot(212)
        plt.imshow(threshold_map.T, cmap='Greys', aspect='auto', extent=self.extent)
        plt.title("(b)", loc='left')
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Range (m)')
        plt.colorbar(label='Intensity Level')
        plt.show()

    def testing(self):
        signal1 = self.tx_signal.generateTXSignal(self.operation)
        self.plotter(signal1, 2)

def main():
    eattack_select = 3
    fc = 9e3
    operation = 0

    obj = ELB780_2(eattack_select, fc, operation)
    obj.signal_construction()
    
    # defines the threads for playing and recording
    ithread = Thread(target=obj.listener)
    othread = Thread(target=obj.emitter)
    # starts the playing and recording threads
    ithread.start()
    othread.start()

    # waits until the output and input threads are finished
    print("Waiting for the playing and recording threads to finish...")
    while othread.is_alive() or ithread.is_alive():
        time.sleep(0.25)

    obj.terminate_pa()
    obj.pulse_compression()
    obj.data_matrix()

    obj.doppler_processing()
    #obj.cfar2Ddetection(3, 5, 20.745)

if __name__ == '__main__':
    main()