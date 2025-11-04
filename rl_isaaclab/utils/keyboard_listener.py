import threading
import time

from pynput import keyboard

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self, keyboard_listen_flag: ThreadSafeValue):
        self.keyboard_listen_flag = keyboard_listen_flag
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _on_press(self, key):
        try:
            if key.char == 'q':
                print('[Keyboard] Moving home.')
                self.keyboard_listen_flag.set(2)
            elif key.char == 'w':
                print('[Keyboard] Freeze actions.')
                self.keyboard_listen_flag.set(3)
            elif key.char == 'e':
                print('[Keyboard] Start policy.')
                self.keyboard_listen_flag.set(4)
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
