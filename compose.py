import mido
import threading
import queue
import time

class Controller:
    DEFAULT_PORT_NAME = "Axiom Pro 49"
    def __init__(self, port=None):
        self._init_port(port)
        self.q = queue.Queue()
        self.filters = { }
        self._running = False
        self._filter_id = 0
        self.notes_on = { }
    
    def _init_port(self, port):
        if port is None:
            port = [i for i in mido.get_input_names()
                    if self.DEFAULT_PORT_NAME.lower() in i.lower()][0]
            port = mido.open_input(port)
        print('Connected to port', port.name)
        self.port = port

    def register_filter(self, filt):
       self._filter_id += 1
       self.filters[self._filter_id] = filt
       return self._filter_id
       
    def remove_filter(self, filt_id):
        filt = self.filters.get(filt_id)
        if filt:
            del self.filters[filt_id]
        return True if filt else False 
        
    def _filter(self, msg):
        return all(f(msg) for f in self.filters.values()) 
        
    def stop(self):
        self._running = False
        
    def get(self):
        return self.q.get() 
        
    def iter_msgs(self):
        while not self.q.empty():
            yield self.q.get()

    def _clear_port(self):
        list(self.port.iter_pending())
        
    def _process(self, msg):
        if msg.type == 'note_on':
            self.notes_on[msg.note] = msg
        elif msg.type == 'note_off':
            if self.notes_on.get(msg.note):
                self.notes_on.pop(msg.note)
        print(self.notes_on.keys()) # test
        
    def listen(self, types=['note_on','note_off']):
        self.types = types
        self.register_filter(lambda msg: msg.type in self.types)
        t = threading.Thread(target=self._listen)
        t.start()
         
    def _listen(self):
        self._running = True
        self._clear_port()
        self.t0 = time.time()
        while self._running:
            msg = self.port.receive()
            if self._filter(msg):
                msg.__dict__['t'] = time.time() - self.t0
                self.q.put(msg) 
                self._process(msg) # test
                #self._process(self.q.get()) # test
