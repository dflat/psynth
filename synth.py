import threading
import math
import numpy as np
import sounddevice as sd
import time
from enum import IntFlag
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

class Status(IntFlag):
    ON = 1
    ATTACKING = 2
    RELEASING = 4

class Note:
    MIDI_TO_FREQ = {midi_val:midi_to_freq(midi_val) for midi_val in range(128)}

    def __init__(self, midi_val, start_frame):
        self.midi_val = midi_val
        self.freq = self.MIDI_TO_FREQ[midi_val]
        self.start = start_frame
        self.status = Status.ON
        self.released_at = False
        self._hash = hash((midi_val,self.start)) #hash((self.midi_val, self.status))#self.start))

    def release(self):
        self.status |= Status.RELEASING

    def is_releasing(self):
        return self.status & Status.RELEASING

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        s = "Note(midi={}, start={}, status={})"
        return s.format(self.midi_val, self.start, repr(self.status))

    @classmethod
    def from_msg(cls, msg, start_frame):
        return cls(msg.note, start_frame)


class Synth:
    
    def __init__(self, controller):
        self.controller = controller
        self.channels = 1
        self.dtype = np.int16
        self.MAX_A = 2**16/2 - 1
        self.CEIL_A = int(0.25*self.MAX_A)
        self.A = self.MAX_A/8
        self.sr = 44100
        self.frame_index = 0
        self.phase = 0
        self.event = threading.Event()
        self.notes_on = { } # map Note => time note started
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
        self.frame_index = 0
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

    def _apply_attack_envelope(self, wave, t, note):
        """ if note is just starting, ramp up its amplitude """
        playhead_t = self.frame_index # current sample index
        #note_t0 = self.notes_on[note]
        note_t0 = note.start
        dt = playhead_t - note_t0 # dt is how far (in samples) into note's ASDR cycle
        print(f'note:{note}, dt:{dt}')
        if dt < self.attack_samples: #self.attack_t*self.sr:#*2:
            # align envelope with wave
            wave *= self._attack_envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        return wave

    def _is_releasing(self, note):
        return note in self.releasing

    def _release_note(self, note):
        note.release()
        self.releasing[note] = self.frame_index # log sample-index at which release triggered
        #del self.notes_on[note]  # TESTING!
        print('releasing note:', note)

    def _apply_release_envelope(self, wave, t, note):
        playhead_t = self.frame_index # current sample index
        released_t0 = self.releasing[note]
        dt = playhead_t - released_t0 # dt is how far (in samples) into note's release
        print(f'releasing note:{note}, dt:{dt}')
        wave *= self._release_envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        if dt > self.release_samples: #self.release_t*self.sr:
            self._remove_note(note)
        return wave

    def _remove_note(self, note):
        #if note in self.notes_on:
        #    del self.notes_on[note] 
        if note in self.releasing:
            del self.releasing[note]
        print(f'removed note:{note}')

    def _start_note(self, note, t0):
        self.notes_on[note] = t0 # value of starting t-index in discrete sample domain 
        print(f'started note: {note} at sample: {t0}')

    def _is_new_note(self, note):
        return note not in self.notes_on

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

    def _get_held_notes(self):
        return [Note.from_msg(msg, self.frame_index) for msg in self.controller.notes_on.values()]

    def _get_sounding_notes(self):
        notes_held = self._get_held_notes()
        #notes_held = self.notes_on   ### TESTING, maybe use above line instead if necessary 
        notes_releasing = set(self.releasing)
        #return notes_held + list(notes_releasing)
        return set(notes_held) | notes_releasing

    def _update_notes_on(self):
        """ keep internal note-state dict in sync with controller's note state """
        to_drop = set(self.notes_on) - set(self._get_held_notes())
        to_drop -= set(self.releasing) # don't redrop currently releasing notes
        for note in to_drop:
            self._release_note(note)

    def _start_note2(self, note):
        self.notes_on[note.midi_val] = note
        print(f'started note: {note}')

    def _release_note2(self, midi_val):
        note = self.notes_on.pop(midi_val)
        if note:
            note.release()
            note.released_at = self.frame_index
            self.releasing[note] = self.frame_index # todo: check for bug
            print('releasing note:', note)

    def _sounding_notes2(self):
        notes_held = set(self.notes_on.values())
        notes_releasing = set(self.releasing)
        return notes_held | notes_releasing

    def _get_wave2(self, t):
        zero_wave = 0*t
        waves = [zero_wave]
        for msg in self.controller.iter_msgs():
            if msg.type == 'note_on':
                note = Note.from_msg(msg, self.frame_index)
                self._start_note2(note)  
            elif msg.type == 'note_off':
                self._release_note2(msg.note)

        notes = self._sounding_notes2()
        n = len(notes)
        for note in notes:
            w = 2*np.pi*note.freq
            wave = (self.A)*self._wave_func(t, w)
            wave = self._apply_attack_envelope(wave, t, note)
            if note.is_releasing():
                wave = self._apply_release_envelope(wave, t, note)
            waves.append(wave)
        return self._sum_waves(waves)
        
    def _get_wave(self, t):
        zero_wave = 0*t
        waves = [zero_wave]
        notes_in_chord = self._get_sounding_notes()
        n = len(notes_in_chord)
        for note in notes_in_chord: 
            if self._is_new_note(note):
                self._start_note(note, self.frame_index)  # probaby should wrap in a Note class
            w = 2*np.pi*note.freq
            #wave = (self.A/n)*self._wave_func(t, w)
            wave = (self.A)*self._wave_func(t, w)
            wave = self._apply_attack_envelope(wave, t, note)
            if self._is_releasing(note):
                wave = self._apply_release_envelope(wave, t, note)
            waves.append(wave)
        return self._sum_waves(waves)

    def callback(self, outdata, frames, ctime, status):
            if status:
                print(status, file=sys.stderr)

            #self._update_notes_on()

            t = (self.frame_index + np.arange(frames)) / self.sr
            t = t.reshape(-1, 1)
            #wave = self._get_wave(t)
            wave = self._get_wave2(t)

            outdata[:] = wave
            self.frame_index += frames
            #print(f'frames:{frames}\ttime_index:{t[0,0]}')
            #self.ctime = ctime
            #print(f'frames:{frames}\ttime_index:{t[0,0]}, ctime:{ctime}')
class Enveloper:
    def __init__(self, a, s, d, r):
        self.attack = a
        self.sustain = s
        self.decay = d
        self.release = r
