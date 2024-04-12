from ea_es import TXSignal
import numpy as np
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

    def terminate_pa( self ):
        '''Terminates the pyaudio binding to portaudio'''
        self.p.terminate()

    def pulse_compression(self):
        '''Applies the matched filter to the received data'''
        print("Applying the matched filter")
        self.tx_signal.plotter(self.rcv, 1)
        self.mfiltered = np.convolve(
            hilbert(self.rcv), 
            np.conj(np.flipud(hilbert(self.tx_signal.pulse))), 
            mode='same'
        )
        self.tx_signal.plotter(self.mfiltered.real, 2)

    def find_1st_peak(self):
        stop_search = int(self.mfiltered.size / 2)
        peaks, _ = find_peaks(np.abs(self.mfiltered[:stop_search]), height=10)
        trace_peak = 0

        # run through the peaks to find the difference in samples between consecutive peaks
        for i in range(peaks.size):
            distance = peaks[i + 1] - peaks[i]
            if distance > 25:
                trace_peak = i
                break
        
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
        self.tx_signal.plotter(analytic_sigg, 2)

    def signal_construction(self):
        '''Constructs the waveform to survey the environment'''
        esupport_sig = self.tx_signal.generateTXSignal(self.operation)
        smps = np.arange(esupport_sig.size)
        plt.subplot(111)

        # construct the bytes signal based on the value of the eattack_select variable
        if self.eattack_select == 1:         # range gate
            eattack_sig = self.tx_signal.generate_rgpo_sig()
            plt.plot(smps, eattack_sig)
            output_bytes = np.column_stack((esupport_sig, eattack_sig))
            self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()
        elif self.eattack_select == 2:       # velocity gate pull-off
            pass
        elif self.eattack_select == 3:       # cover pulse
            pass
        elif self.eattack_select == 4:       # multiple false targets
            pass
        else:
            self.output_bytes = (self.tx_signal.volume * esupport_sig).tobytes()

        plt.plot(smps, esupport_sig)
        plt.xlabel('samples')
        plt.ylabel('amplitude')
        plt.grid()
        plt.show()

    def testing(self):
        signal1 = self.tx_signal.generateTXSignal(self.operation)
        self.plotter(signal1, 2)

def main():
    eattack_select = 1
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

if __name__ == '__main__':
    main()