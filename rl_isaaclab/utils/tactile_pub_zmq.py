import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add the tactile_sdk directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tactile_sdk'))

from tactile_sdk.sharpa.tactile import Touch, TouchSetting
import logging
import numpy as np
import subprocess
import re

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
import time
import argparse
import struct
import importlib.util
import threading
from types import SimpleNamespace
import sharpa_pb2 as pb
from zmq_wrapper import ZmqWrapper
import zmq
from collections import deque

hands = ['L', 'R']
HOST_IP = '192.168.10.240'
CHANNELS = []
PORTS = []
if 'L' in hands:
    CHANNELS.append(range(5,10))
    PORTS.append(50011)
if 'R' in hands:
    CHANNELS.append(range(0,5))
    PORTS.append(50001)

class WallTimer:
    def __init__(self, period_sec, callback):
        self.period = period_sec
        self.callback = callback
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        while self._running:
            start_time = time.time()
            self.callback()
            elapsed = time.time() - start_time
            sleep_time = max(0, self.period - elapsed)
            time.sleep(sleep_time)


class Counter:
    def __init__(self, name, period=1.0):
        self.period = period
        self.name = name
        self.stamp = time.time()
        self.counter = 0
        self.reset()

    def timely_report(self):
        now = time.time()
        if now - self.stamp <= self.period:
            return

        metrics = {
            'tactile_R': (
                lambda: ','.join(str(item / self.period) for item in self.tactile[:5])\
                    +'] temprature:[' + ','.join('{:.3f}'.format(item) for item in self.temperature[:5]),
                sum(self.tactile[:4])
            ),
            'tactile_L': (
                lambda: ','.join(str(item / self.period) for item in self.tactile[5:])\
                    +'] temprature:[' + ','.join('{:.3f}'.format(item) for item in self.temperature[5:]),
                sum(self.tactile[5:])
            ),
        }
        report_items = [
            f"{name}[{value_func()}]" 
            for name, (value_func, condition) in metrics.items() 
            if condition
        ]
        if report_items:
            logging.info(f"{self.name}: {', '.join(report_items)}")

        self.stamp = now
        self.reset()

    def reset(self):
        self.tactile = [0] * 10
        self.temperature = [0] * 10

def get_ch_name(ch):
    ch_name=['10005', '10006', '10007', '10008', '10009',  # right
             '10015', '10016', '10017', '10018', '10019']  # left
    return ch_name[ch]

class TactilePublisherZmq:
    def __init__(self):
        self.ct = Counter('tactile', 1.0)
        self.hs_touch_t0 = time.perf_counter()
        self.zmq_pub = ZmqWrapper(zmq.PUB, "tcp://*:48006")
        self.data = {}
        self.period = 1.0
        self.stamp = 0.0
        self.data = { ch: (None, None, None, None) for ch in range(10) }
        self.timer = WallTimer(1.0/30.0, self.timer_callback)
        self.timer.start()

    def pub(self, ch, ts, raw=None, f6=None, deform=None, temperature=None):
        self.ct.tactile[ch] += 1
        self.data[ch] = (ts, raw, f6, deform)
    
    def timer_callback(self):
        self.ct.timely_report()
        for ch, (ts, raw, f6, deform) in self.data.items():
            if ts is None or raw is None or f6 is None or deform is None:
                continue

            if len(f6) == 0 or len(deform) == 0:
                print("Error: f6 or deform is empty")
                return
            try:
                # 打包数据
                # packed_data = struct.pack(format_string, *f6, *deform, ch)
                tactile_msg = pb.Tactile()
                tactile_msg.deform.height, tactile_msg.deform.width = 240, 240
                tactile_msg.header.stamp.sec = int(ts)
                tactile_msg.header.stamp.nanosec = int((ts - int(ts))* 1e9)
                tactile_msg.header.key = get_ch_name(ch)
                tactile_msg.deform.data = deform.tobytes()
                tactile_msg.force6d.force.x = f6[0]
                tactile_msg.force6d.force.y = f6[1]
                tactile_msg.force6d.force.z = f6[2]
                tactile_msg.force6d.torque.x = f6[3]
                tactile_msg.force6d.torque.y = f6[4]
                tactile_msg.force6d.torque.z = f6[5]
                # 发送数据
                msg = tactile_msg.SerializeToString()
                self.zmq_pub.send(msg)
                # print(f'send msg len={len(msg)}')
            except struct.error as e:
                print(f"Error in struct.pack: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                logging.error(traceback.format_exc())   


try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from geometry_msgs.msg import WrenchStamped, TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from cv_bridge import CvBridge
except ImportError:
    class Node:
        def __init__(self, name):
            self.name = name

    class Header:
        def __init__(self):
            self.stamp = 0
            self.frame_id = ''
    
    class CvBridge:
        def cv2_to_imgmsg(self, img, encoding='mono8', header=None):
            return None

class TactilePublisher(Node):
    def __init__(self, buffer_size=2, fake_pose=False):
        super().__init__('tactile')
        bz = buffer_size
        channel = range(10)
        self.pub_raw = {ch: self.create_publisher(Image, f'tactile/ch_{ch}/raw', bz) for ch in channel}
        self.pub_f6 = {ch: self.create_publisher(WrenchStamped, f'tactile/ch_{ch}/force6d', bz) for ch in channel}
        self.pub_deform = {ch: self.create_publisher(Image, f'tactile/ch_{ch}/deform', bz) for ch in channel}
        # record start time
        self.hs_touch_t0 = time.perf_counter()
        self.ros_t0 = self.get_clock().now()

    def pub(self, ch, ts, raw=None, f6=None, deform=None):
        elapsed_time = ts - self.hs_touch_t0
        current_time = self.ros_t0 + rclpy.time.Duration(seconds=elapsed_time)
        header = Header()
        header.stamp = current_time.to_msg()
        header.frame_id = f'tactile/p_{ch}'
        if raw is not None:
            msg = self.bridge.cv2_to_imgmsg(raw, encoding='mono8', header=header)
            self.pub_raw[ch].publish(msg)
        if f6 is not None:
            msg = WrenchStamped(header=header)
            f6 = f6.flatten().astype(float)
            msg.wrench.force.x = f6[0]
            msg.wrench.force.y = f6[1]
            msg.wrench.force.z = f6[2]
            msg.wrench.torque.x = f6[3]
            msg.wrench.torque.y = f6[4]
            msg.wrench.torque.z = f6[5]
            self.pub_f6[ch].publish(msg)
        if deform is not None:
            msg = Image()
            msg.header = header
            msg.height, msg.width = 240, 240
            msg.encoding = 'mono8'
            msg.is_bigendian = False
            msg.step = 240 * 1
            msg.data = deform.tobytes()
            self.pub_deform[ch].publish(msg)
        

class TactileRunner:
    def __init__(self, enable_ros=False, enable_zmq=True):
        try:
            self.tac_ros = None
            self.tac_zmq = None
            self.frame_count = 0
            self.touch = []  # Initialize touch list
            self.__start(enable_ros, enable_zmq)
        except Exception as e:
            logging.error(f'tactile node init failed: {e}')
            import traceback
            logging.error(traceback.format_exc())
            raise e

    def __callback(self, frame):
        ch = frame['channel']
        ts = frame['ts']
        img, f6, deform = None, None, None    
        if frame['content'].get("RAW") is not None: img = frame['content']['RAW'].squeeze()
        if frame['content'].get("DEFORM") is not None: deform = frame['content']['DEFORM'].squeeze()
        if frame['content'].get('F6') is not None: f6 = frame['content']['F6'].squeeze()
        self.frame_count += 1
        if self.tac_zmq:
            self.tac_zmq.pub(ch, ts, img, f6, deform)
        if self.tac_ros:
            self.tac_ros.pub(ch, ts, img, f6, deform)

    def __start(self, enable_ros=False, enable_zmq=True):
        if enable_ros:
            rclpy.init(args=sys.argv)
            self.tac_ros = TactilePublisher()
            logging.info(f'tactile ros node started')
        if enable_zmq:
            self.tac_zmq = TactilePublisherZmq()
            logging.info(f'tactile zmq node started')
        
        try:
            for CHANNEL, PORT in zip(CHANNELS, PORTS):
                setting = TouchSetting(
                    model_path={"/home/sharpa/tactile_ws/tactile_sdk/config/models/EF_lighter128bias_0905_200d5_DB_DLV15.onnx": CHANNEL},
                    infer_from_device=False)
                self.touch.append(Touch(HOST_IP, PORT, CHANNEL, callback=self.__callback, setting=setting))
                logging.info(f'Touch object created successfully')
                logging.info(f'Starting Touch with HOST_IP={HOST_IP}, HOST_PORT={PORT}, CHANNEL={CHANNEL}')
                if self.touch[-1].start():
                    logging.info(f'tactile touch start success')
        except Exception as e:
            logging.error(f'tactile touch start failed: {e}')
            import traceback
            logging.error(traceback.format_exc())
        
    def run(self):
        print("TactileRunner.run() called")
        if self.tac_ros:
            logging.info(f'tactile ros node started')
            rclpy.spin(self.tac_ros)
            for touch in self.touch:
                touch.stop()
            self.tac_ros.destroy_node()
            rclpy.shutdown()
        else:
            logging.info(f'tactile zmq node started')
            print("Entering ZMQ main loop...")
            try:
                frame_count = 0
                last_time = time.time()
                while True:
                    time.sleep(0.1)
                    # Add some periodic logging to show the program is still running
                    current_time = time.time()
                    if current_time - last_time > 5.0:  # Log every 5 seconds
                        print(f"ZMQ node running, (total frames received: {self.frame_count})")
                        # logging.info(f"ZMQ node running, waiting for data... (total frames received: {self.frame_count})")
                        # Try to check if Touch object is still running
                        if hasattr(self.touch, 'is_running'):
                            print(f"Touch is_running: {self.touch.is_running()}")
                        if hasattr(self.touch, 'get_status'):
                            print(f"Touch status: {self.touch.get_status()}")
                        last_time = current_time
            except KeyboardInterrupt:
                logging.info("Received keyboard interrupt, shutting down...")
                for touch in self.touch:
                    touch.stop()
                if self.tac_zmq:
                    self.tac_zmq.timer.stop()


def run(args):
    print("Creating TactileRunner...")
    runner = TactileRunner()
    print("Starting runner.run()...")
    runner.run()


def main():
    parser = argparse.ArgumentParser(description='tactile publisher')
    parser.add_argument('--cfg', type=str, help='load config', default='')
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
