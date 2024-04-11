from ea_esm import TXSignal
import numpy as np
from matplotlib import pyplot as plt
import pyaudio, queue, time
from threading import Thread
from scipy.signal import hilbert

class ELB780_2:
    def __init__(self, esm_select, ea_select, fc, operation):
        '''Initialises the project'''
        self.operation = operation
        self.chunk = 1024
        self.rcv = []                               # recorded echo data
        self.tx_signal = TXSignal(esm_select, ea_select, fc)

        # pyAudio params
        self.p = pyaudio.PyAudio()                  # Create an interface to PortAudio
        self.format = pyaudio.paFloat32
        self.channel = 1 if self.tx_signal.ea_select == 0 else 2

        plt.close('all')

    def emitter(self):
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
        """Terminates the pyaudio binding to portaudio"""
        self.p.terminate()

    def pulse_compression(self):
        print("Applying the matched filter")
        self.tx_signal.plotter(self.rcv, 1)
        self.mfiltered = np.convolve(
            hilbert(self.rcv), 
            np.conj(np.flipud(hilbert(self.tx_signal.pulse))), 
            mode='same'
        )
        self.tx_signal.plotter(self.mfiltered.real, 2)

    def signal_construction(self):
        esm_sig = self.tx_signal.generateTXSignal(self.operation)
        ea_sig = self.tx_signal.generate_rgpo_sig()
        #self.tx_signal.plotter(output_bytes, 2)

        output_bytes = np.column_stack((esm_sig, ea_sig))
        self.output_bytes = (self.tx_signal.volume * output_bytes).tobytes()

        smps = np.arange(esm_sig.size)

        '''plt.subplot(111)
        plt.plot(smps, esm_sig)
        plt.plot(smps, ea_sig)
        plt.xlabel('samples')
        plt.ylabel('amplitude')
        plt.grid()
        plt.show()'''

    def testing(self):
        signal1 = self.tx_signal.generateTXSignal(self.operation)
        self.plotter(signal1, 2)

def main():
    obj = ELB780_2(esm_select=0, ea_select=0, fc=8e3, operation=0)
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

if __name__ == '__main__':
    main()