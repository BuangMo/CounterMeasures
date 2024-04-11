import pyaudio, time
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt

class TwoSpeakers:
    def __init__(self, fs, duration):
        self.Fs = fs
        self.duration = duration

        # pyAudio params
        self.p = pyaudio.PyAudio()                  # Create an interface to PortAudio
        self.format = pyaudio.paFloat32
        self.channel = 2
        self.chunk = 1024

    def generate_waveforms(self, f0, f1):
        t = np.linspace(0, self.duration, int(self.duration * self.Fs), endpoint=False)
        square_waveform0 = np.ones(len(t))
        square_waveform1 = np.ones(len(t))
        square_waveform0[0:24000] = 0
        square_waveform1[24001:self.Fs] = 0
        sig0 = np.sin(2 * np.pi * f0 * t) * square_waveform0
        sig1 = np.sin(2 * np.pi * f1 * t) * square_waveform1

        self.output_bytes = np.column_stack((sig0, sig1)).astype(np.float32).tobytes()

        #plt.plot(t, square_waveform1)
        #plt.show()

    def emitter(self):
        ostream = self.p.open(
            format=self.format,
            channels=self.channel,
            rate=int(self.Fs),
            output=True,
            frames_per_buffer=self.chunk
        )

        start_time = time.time()
        print("Playing...")
        ostream.write(self.output_bytes)
        print("Played sound for {:.2f} seconds".format(time.time() - start_time))

        # Stop and Close the output stream
        ostream.stop_stream()
        ostream.close()

    def terminate_pa(self):
        """Terminates the pyaudio binding to portaudio"""
        self.p.terminate()

def main():
    obj = TwoSpeakers(48000, 1)
    obj.generate_waveforms(1000.0, 9000.0)
    
    # defines the threads for playing and recording
    othread = Thread(target=obj.emitter)
    # starts the playing and recording threads
    othread.start()
    time.sleep(3)
    obj.terminate_pa

if __name__ == "__main__":
    main()
