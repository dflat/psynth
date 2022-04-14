import threading
import math
import numpy as np
import sounddevice as sd
import time
from copy import deepcopy
import matplotlib.pyplot as plt

#fig, ax = plt.subplots()

def plot(X):
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
        self.MAX_A = 2**16/2 - 1
        self.CEIL_A = int(0.25*self.MAX_A)
        self.A = self.MAX_A/8
        self.freq = 440
        self.sr = 44100
        self.index = 0
        self.phase = 0
        self.event = threading.Event()
        self.notes_on = { } # map midi_note => time note started
        self.releasing = { }
        self._running = False
        self.config = dict(a=1,b=1,c=0)
        self._set_envelope() # testing
        
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
        #self._set_envelope() # testing
        self.index = 0
        with sd.OutputStream(channels=self.channels,
                             callback=lambda o,f,t,s: self.callback(o,f,t,s),
                             samplerate=self.sr,
                             dtype=self.dtype):
            print('playing..')
            self.event.wait()
            print('stopped')
        
    def _set_envelope(self):
        self._set_attack_envelope(.2)
        self._set_release_envelope(.2)

    @property
    def release(self):
        return self.release_t
    @release.setter
    def release(self, release_time):
        self._set_release_envelope(release_time)

    @property
    def attack(self):
        return self.attack_t
    @attack.setter
    def attack(self, attack_time):
        self._set_attack_envelope(attack_time)

    def _set_release_envelope(self, release_time):
        self.ASDR_t = 6
        self.release_t = release_time
        release_samples = int(self.release_t*self.sr)
        self.release_samples = release_samples
        n = self.sr*self.ASDR_t
        T = np.arange(n) / self.sr
        end_t = T[release_samples]
        base = np.zeros(len(T))
        base[:release_samples] = np.cos(np.pi/2 * T[:release_samples]/end_t)
        self._release_envelope = base.reshape(-1,1) 

    def _set_attack_envelope(self, attack_time):
        self.ASDR_t = 6
        self.attack_t = attack_time
        attack_samples = int(self.attack_t*self.sr)
        self.attack_samples = attack_samples
        n = self.sr*self.ASDR_t
        T = np.arange(n) / self.sr
        end_t = T[attack_samples]
        base = np.ones(len(T))
        base[:attack_samples] = np.sin(np.pi/2 * T[:attack_samples]/end_t)
        self._attack_envelope = base.reshape(-1,1) 

    def _apply_attack_envelope(self, wave, t, midi_note):
        """ if note is just starting, ramp up its amplitude """
        playhead_t = self.index # current sample index
        note_t0 = self.notes_on[midi_note]
        dt = playhead_t - note_t0 # dt is how far (in samples) into note's ASDR cycle
        print(f'midi_note:{midi_note}, dt:{dt}')
        if dt < self.attack_samples: #self.attack_t*self.sr:#*2:
            # align envelope with wave
            wave *= self._attack_envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        return wave

    def _update_notes_on(self):
        """ keep internal note-state dict in sync with controller's note state """
        to_drop = set(self.notes_on) - set(self.controller.notes_on)
        to_drop -= set(self.releasing) # don't redrop currently releasing notes
        for note in to_drop:
            self._release_note(note)

    def _is_releasing(self, midi_note):
        return midi_note in self.releasing

    def _release_note(self, midi_note):
        self.releasing[midi_note] = self.index # log sample-index at which release triggered
        print('releasing note:', midi_note)

    def _apply_release_envelope(self, wave, t, midi_note):
        playhead_t = self.index # current sample index
        released_t0 = self.releasing[midi_note]
        dt = playhead_t - released_t0 # dt is how far (in samples) into note's release
        print(f'releasing midi_note:{midi_note}, dt:{dt}')
        wave *= self._release_envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        if dt > self.release_samples: #self.release_t*self.sr:
            self._remove_note(midi_note)
        return wave

    def _remove_note(self, midi_note):
        if midi_note in self.notes_on:
            del self.notes_on[midi_note] 
        if midi_note in self.releasing:
            del self.releasing[midi_note]
        print(f'removed midi_note:{midi_note}')

    def _start_note(self, midi_note, t0):
        # todo: apply attack curve
        self.notes_on[midi_note] = t0 # value of starting t-index in discrete sample domain 
        print(f'started note: {midi_note} at sample: {t0}')

    def _is_new_note(self, midi_note):
        return midi_note not in self.notes_on

    def _wave_func(self, t, w):
        a,b,c = self.config.values() # modulate wave shape with parameters
        wave = np.sin(c*np.sin(b*w*t) + a*w*t)
        return wave

    def _sum_waves(self, waves):
        mix = sum(waves[1:], start=waves[0])
        max_a = mix.max()
        #if max_a > self.CEIL_A: # clamp values to ceiling amplitude
        #print(f'before -- max:{mix.max():.2f}, min:{mix.min():.2f}')
        #mix = mix - (mix**2)/(self.CEIL_A*2) #self.CEIL_A*(mix/max_a)
        #print(f'after -- max:{mix.max():.2f}, min:{mix.min():.2f}')
        return mix

    def _get_sounding_notes(self):
        notes_held = deepcopy(self.controller.notes_on)
        notes_releasing = set(self.releasing)
        return set(notes_held) | notes_releasing

    def _get_wave(self, t):
        zero_wave = 0*t
        waves = [zero_wave]
        n = len(self.controller.notes_on) # compress amplitude proportional to # of notes in chord
        #notes_in_chord = deepcopy(self.controller.notes_on)
        notes_in_chord = self._get_sounding_notes()
        #for midi_note, msg in notes_in_chord.items(): # build each wave in the chord
        for midi_note in notes_in_chord: # build each wave in the chord
            if self._is_new_note(midi_note):
                self._start_note(midi_note, self.index)  # probaby should wrap in a Note class
            freq = self.MIDI_TO_FREQ[midi_note]
            w = 2*np.pi*freq             # radians/sec (or 'angular frequency')
            #wave = (self.A/n)*self._wave_func(t, w)
            wave = (self.A)*self._wave_func(t, w)
            wave = self._apply_attack_envelope(wave, t, midi_note)
            if self._is_releasing(midi_note):
                wave = self._apply_release_envelope(wave, t, midi_note)
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
class Note:
    def __init__(self, midi_val, start_sample):
        self.midi_val = midi_val
        self.start = start_sample
        self.status = 'on'
    def release(self):
        self.status = 'releasing'

