import threading
import time

from pynput import keyboard

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self, 
                 deploy_state_flag: ThreadSafeValue, 
                 calib_tactile_flag: ThreadSafeValue, 
                 hand_ip: str):
        self.deploy_state_flag = deploy_state_flag
        self.hand_ip = hand_ip
        self.calib_tactile_flag = calib_tactile_flag
        self.last_deploy_state_flag = self.deploy_state_flag.get()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _on_press(self, key):
        '''
        deploy_state_flag:
            0: move home
            1: freeze actions
            2: reset env
            3: run policy
        calib_tactile_flag:
            0: silence
            1: calibrate tactile
        '''
        try:
            if key.char == 'q':
                print('[Keyboard] Moving home.')
                self.deploy_state_flag.set(0)
            elif key.char == 'w':
                if self.deploy_state_flag.get() != 1:
                    self.last_deploy_state_flag = self.deploy_state_flag.get()
                    print('[Keyboard] Freeze actions.')
                    self.deploy_state_flag.set(1)
                else:
                    print('[Keyboard] Continue actions.')
                    self.deploy_state_flag.set(self.last_deploy_state_flag)
            elif key.char == 'e':
                print('[Keyboard] Start policy.')
                self.deploy_state_flag.set(2)
            elif key.char == 't':
                self.calib_tactile_flag.set(1)
                print("[Keyboard] Tactile calibration.")
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
