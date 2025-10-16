import threading
import time

from pynput import keyboard

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self, saving_flag: ThreadSafeValue):
        self.saving_flag = saving_flag
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _on_press(self, key):
        try:
            if key.char == 's':
                print("[Keyboard] Start saving data.")
                self.saving_flag.set(1)
            elif key.char == 'd':
                print("[Keyboard] Stop saving and write to file.")
                self.saving_flag.set(0)
            else:
                pass
        except:
            pass

    def _run(self):
        with keyboard.Listener(on_press=self._on_press) as listener:
            while not self.stop_event.is_set():
                time.sleep(0.01)
            listener.stop()
        print("[Keyboard] Keyboard listener stopped.")
    
    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
