import threading
import math
import numpy as np
import sounddevice as sd
import time
from copy import deepcopy
import matplotlib.pyplot as plt

def plot(X):
    fig, ax = plt.subplots()
    ax.plot(X)

# helpers
def midi_to_freq(midi_note):
    A = 220 # freq for midi value 57
    diff = midi_note - 57
    return A*2**(diff/12)

class Synth:
    MIDI_TO_FREQ = {note:midi_to_freq(note) for note in range(128)}
    
    def __init__(self, controller):
        self.controller = controller
        self.channels = 1
        self.dtype = np.int16
        self.MAX_A = 2**16 - 1
        self.A = self.MAX_A/8
        self.freq = 440
        self.sr = 44100
        self.index = 0
        self.phase = 0
        self.event = threading.Event()
        self.notes_on = { } # map midi_note => time note started
        self._running = False
        self.config = dict(a=1,b=1,c=0)
        
    def stop(self):
        self.event.set()
        self.controller.stop()
        self._running = False
    
    def play(self):
        self.controller.listen()
        self.event.clear()
        self._running = True
        t = threading.Thread(target=self._play)
        t.start()
        
    def _play(self):
        self.t0 = time.time()
        self._get_envelope() # testing
        self.index = 0
        with sd.OutputStream(channels=self.channels,
                             callback=lambda o,f,t,s: self.callback(o,f,t,s),
                             samplerate=self.sr,
                             dtype=self.dtype):
            print('playing..')
            self.event.wait()
            print('stopped')
        
    def _get_envelope(self):
        self.ASDR_t = 12
        self.attack_t = 1
        attack_samples = self.attack_t*self.sr
        n = self.sr*self.ASDR_t
        T = np.arange(n) / self.sr
        end_t = T[attack_samples]
        base = np.ones(len(T))#1*T
        base[:attack_samples] = np.sin(np.pi/2 * T[:attack_samples]/end_t)

        self._envelope = base.reshape(-1,1) # + whatever
        #plot(self._envelope)
        print(self._envelope)
        # todo: pregenerate envelope and then apply to each note?

    def _apply_envelope(self, wave, t, midi_note):
        #playhead_t = t[0,0] # maybe use t-index (samples) instead of pre-converting to seconds
        playhead_t = self.index 
        note_t0 = self.notes_on[midi_note]
        dt = playhead_t - note_t0 # dt is how far (in samples) into note's ASDR cycle
        if dt < self.attack_t*self.sr:#*2:
            # align envelope with wave
            wave *= self._envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        return wave

    def _apply_attack_curve(self, wave):
        """ if note is just starting, ramp up its amplitude """
        #...todo
        return wave

    def _update_notes_on(self):
        """ keep internal note-state dict in sync with controller's note state """
        to_drop = set(self.notes_on) - set(self.controller.notes_on)
        for key in to_drop:
            self._release_note(key)

    def _release_note(self, midi_note):
        # todo: apply release curve
        del self.notes_on[midi_note] # note: maybe do this, here or maybe just 'mark for deletion'
        print('released note:', midi_note)

    def _start_note(self, midi_note, t0):
        # todo: apply attack curve
        self.notes_on[midi_note] = t0 # value of starting t-index in discrete sample domain 
        print(f'started note: {midi_note} at time: {t0:.2f}')

    def _is_new_note(self, midi_note):
        return midi_note not in self.notes_on

    def _wave_func(self, t, w):
        a,b,c = self.config.values() # modulate wave shape with parameters
        wave = np.sin(c*np.sin(b*w*t) + a*w*t)
        return wave

    def _sum_waves(self, waves):
        return sum(waves[1:], start=waves[0])

    def _get_wave(self, t):
        zero_wave = 0*t#np.sin(t)
        n = len(self.controller.notes_on) # compress amplitude proportional to # of notes in chord
        notes_in_chord = deepcopy(self.controller.notes_on)
        waves = [zero_wave]
        for midi_note, msg in notes_in_chord.items(): # build each wave in the chord
            if self._is_new_note(midi_note):
                #self._start_note(midi_note, t[0,0])
                self._start_note(midi_note, self.index) # testing
            freq = self.MIDI_TO_FREQ[midi_note]
            w = 2*np.pi*freq             # radians/sec (or 'angular frequency')
            wave = (self.A/n)*self._wave_func(t, w)
            #wave = self._apply_attack_curve(wave) # note: move this?
            wave = self._apply_envelope(wave, t, midi_note)
            waves.append(wave)
        return self._sum_waves(waves)

    def callback(self, outdata, frames, ctime, status):
            if status:
                print(status, file=sys.stderr)

            self._update_notes_on()

            t = (self.index + np.arange(frames)) / self.sr
            t = t.reshape(-1, 1)
            wave = self._get_wave(t)

            outdata[:] = wave
            self.index += frames
            #print(f'frames:{frames}\ttime_index:{t[0,0]}')
            #self.ctime = ctime
            #print(f'frames:{frames}\ttime_index:{t[0,0]}, ctime:{ctime}')
