import threading
import math
import random
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

class Preset:
    PRESETS = { }
    def __init__(self, name, a, s, d, r, alpha, beta, gamma):
        params = locals()
        params.pop('self')
        for k,v in params.items():
            setattr(self, k, v)
    def __repr__(self):
        return repr(self.__dict__ )

    @classmethod
    def random(cls, s, low=1, high=5):
        vowels = 'aeiou'
        nums = '0123456789'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        name = str(''.join([random.choice(i) for i in (vowels, nums, consonants)]))
        vals = [random.randint(low, high) for i in range(3)]
        a = 2*random.random() + .001
        r = 2*random.random() + .001
        p = Preset(name=name,a=a,s=-1,d=-1,r=r,alpha=vals[0],beta=vals[1],gamma=vals[2])
        presets[name] = p
        print(p)
        s.load_preset(name)

presets = {'hollow':Preset(name='hollow', a=.2, s=-1, d=-1, r=1, alpha=1, beta=2, gamma=1)}

class Note:
    MIDI_TO_FREQ = {midi_val:midi_to_freq(midi_val) for midi_val in range(128)}

    def __init__(self, midi_val, start_frame):
        self.midi_val = midi_val
        self.freq = self.MIDI_TO_FREQ[midi_val]
        self.start = start_frame
        self.status = Status.ON | Status.ATTACKING
        self.released_at = False
        self._hash = hash((midi_val,self.start)) #hash((self.midi_val, self.status))#self.start))

    def release(self):
        self.status |= Status.RELEASING

    def end(self):
        self.status &= ~Status.ON

    def is_ended(self):
        return not self.status & Status.ON

    def is_releasing(self):
        return self.status & Status.RELEASING

    def is_attacking(self):
        return self.status & Status.ATTACKING

    def end_attack(self):
        self.status &= ~Status.ATTACKING

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
        self.wave_config = dict(a=1,b=1,c=0)
        self.enveloper = Enveloper(self)
        #self._set_envelope() # testing
        
    def load_preset(self, name): # this will be broken after refactoring to Enveloper
        p = presets.get(name)
        if not p:
            print('No preset named', name)
        self.wave_config = dict(a=p.alpha,b=p.beta,c=p.gamma)
        self.attack = p.a
        self.release = p.r
        print('Loaded preset:', name)

    def stop(self):
        self.event.set()
        self.controller.stop()
        self._running = False
    
    def play(self):
        if self._running:
            return
        self.controller.listen()
        self.event.clear()
        self._running = True
        t = threading.Thread(target=self._play)
        t.start()
        
    def _play(self):
        self.frame_index = 0
        with sd.OutputStream(channels=self.channels,
                             callback=lambda o,f,t,s: self.callback(o,f,t,s),
                             samplerate=self.sr,
                             dtype=self.dtype):
            print('playing..')
            self.event.wait()
            print('stopped')

    def _wave_func(self, t, w):
        a,b,c = self.wave_config.values() # modulate wave shape with parameters
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

    def _start_note(self, note):
        self.notes_on[note.midi_val] = note
        print(f'started note: {note}')

    def _release_note(self, midi_val):
        note = self.notes_on.pop(midi_val)
        if note:
            note.release()
            note.released_at = self.frame_index
            self.releasing[note] = self.frame_index # todo: check for bug
            print('releasing note:', note)

    def _remove_note(self, note):
        if note in self.releasing:
            del self.releasing[note]
            print(f'removed note:{note}')

    def _sounding_notes(self):
        notes_held = set(self.notes_on.values())
        for note in set(self.releasing): 
            if note.is_ended():        # Enveloper will set note state to off, this cleans up
                self._remove_note(note)
        notes_releasing = set(self.releasing)
        return notes_held | notes_releasing

    def _read_input(self):
        for msg in self.controller.iter_msgs():
            if msg.type == 'note_on':
                note = Note.from_msg(msg, self.frame_index)
                self._start_note(note)  
            elif msg.type == 'note_off':
                self._release_note(msg.note)

    def _get_wave(self, t):
        zero_wave = 0*t
        waves = [zero_wave]
        # note: refactored reading from here into _read_input
        notes = self._sounding_notes()
        n = len(notes)
        for note in notes:
            w = 2*np.pi*note.freq
            wave = (self.A)*self._wave_func(t, w)
            
            # apply processing pipleine to wave (just enveloper for now)
            wave = self.enveloper.process(wave, t, note)

            #if note.is_attacking():   # done in enveloper class now
            #    wave = self._apply_attack_envelope(wave, t, note)
            #if note.is_releasing():
            #    wave = self._apply_release_envelope(wave, t, note)

            waves.append(wave)
        return self._sum_waves(waves)
        
    def callback(self, outdata, frames, ctime, status):
            if status:
                print(status, file=sys.stderr)

            self._read_input()

            t = (self.frame_index + np.arange(frames)) / self.sr
            t = t.reshape(-1, 1)
            wave = self._get_wave(t)

            outdata[:] = wave
            self.frame_index += frames

class Enveloper:
    ASDR = dict(a=0.2, s=-1, d=-1, r=1)  # defaults
    def __init__(self, synth, a=None, s=None, d=None, r=None):
        self.synth = synth
        self.buffer_t = 1
        self.attack = a or self.ASDR['a']
        self.sustain = s or self.ASDR['s']
        self.decay = d or self.ASDR['d']
        self.release = r or self.ASDR['r']

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
   
    def process(self, wave, t, note):
        if note.is_attacking():
            wave = self._apply_attack_envelope(wave, t, note)
        if note.is_releasing():
            wave = self._apply_release_envelope(wave, t, note)
        return wave

    def _apply_attack_envelope(self, wave, t, note):
        playhead_t = self.synth.frame_index       # Current frame index
        dt = playhead_t - note.start              # How far (in frames) into note's ASDR cycle
        if dt <= self.attack_frames: 
            wave *= self._attack_envelope[dt:dt+512]  # 512 = buffersize = len(t) on some axis
        else:
            note.end_attack()
        return wave

    def _apply_release_envelope(self, wave, t, note):
        playhead_t = self.synth.frame_index 
        released_t0 = note.released_at
        dt = playhead_t - released_t0 
        wave *= self._release_envelope[dt:dt+512] 
        if dt > self.release_frames: 
            note.end()  # this communicates with synth to remove note
        return wave

    def _set_release_envelope(self, release_time):
        self.release_t = release_time
        release_frames = int(self.release_t*self.synth.sr)
        self.release_frames = release_frames
        n = self.synth.sr*(release_time + self.buffer_t) 
        T = np.arange(n) / self.synth.sr
        end_t = T[release_frames]
        base = np.zeros(len(T))
        base[:release_frames] = np.cos(np.pi/2 * T[:release_frames]/end_t)
        self._release_envelope = base.reshape(-1,1) 

    def _set_attack_envelope(self, attack_time):
        self.attack_t = attack_time
        attack_frames = int(self.attack_t*self.synth.sr)
        self.attack_frames = attack_frames
        n = self.synth.sr*(attack_time + self.buffer_t) 
        T = np.arange(n) / self.synth.sr
        end_t = T[attack_frames]
        base = np.ones(len(T))
        base[:attack_frames] = np.sin(np.pi/2 * T[:attack_frames]/end_t)
        self._attack_envelope = base.reshape(-1,1) 

