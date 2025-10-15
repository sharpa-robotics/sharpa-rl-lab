import logging
import zmq
import threading
import time
import rl_isaaclab.utils.sharpa_pb2 as pb

class ZmqWrapper:
    def __init__(self, mode, con_str, recv_timeout=30000):
        self.mode = mode
        self.context = zmq.Context()
        self.socket = self.context.socket(mode)
        self.lock = threading.Lock()  # 线程锁，保证send()的线程安全
        self.socket.setsockopt(zmq.LINGER, 0)  # 关闭 socket 时不阻塞
        if mode == zmq.PUB:
            self.socket.bind(con_str)
            logging.info(f'create zmq.PUB :{con_str}')
        elif mode == zmq.SUB:
            self.socket.connect(con_str)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            logging.info(f'create zmq.SUB :{con_str}')
        elif mode == zmq.REP:
            self.socket.bind(con_str)
            logging.info(f'create zmq.REP :{con_str}')
        elif mode == zmq.REQ:
            self.socket.connect(con_str)
            logging.info(f'create zmq.REQ :{con_str}')
        elif mode == zmq.PUSH:
            self.socket.connect(con_str)
            logging.info(f'create zmq.PUSH :{con_str}')
        elif mode == zmq.PULL:
            self.socket.bind(con_str)
            logging.info(f'create zmq.PULL :{con_str}')

    def start_receiver_thread(self, callback):
        self.callback = callback
        self.active = True
        thread = threading.Thread(target=self._receive_messages)
        thread.daemon = True  # Makes the thread exit when the main thread exits
        thread.start()

    def stop_thread(self):
        self.active = False

    def _receive_messages(self):
        while self.active:
            try:
                if self.mode == zmq.SUB or self.mode == zmq.PULL:
                    message = self.socket.recv(flags=zmq.NOBLOCK)  # 使用非阻塞模式
                    self.callback(message)
                elif self.mode == zmq.REP:
                    request = self.socket.recv()  # 使用非阻塞模式
                    response = pb.Response()
                    response.success = False
                    try:
                        self.callback(request, response)
                    except Exception as e:
                        logging.error(f"ZMQ REP Callback error: {e}")
                    self.socket.send(response.SerializeToString())
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Receive error: {e}")
                time.sleep(0.1)

    def send(self, message):
        """ 线程安全的 send 方法 """
        with self.lock:  # 确保只有一个线程能访问 send 方法
            try:
                if self.mode == zmq.REQ:
                    self.socket.send(message)
                    try:
                        return self.socket.recv()
                    except zmq.Again:
                        logging.warning("Request timed out")
                        return None
                elif self.mode == zmq.PUB or self.mode == zmq.PUSH:
                    self.socket.send(message)
                    return None
            except Exception as e:
                logging.error(f"Send error: {e}")
                return None

    def close(self):
        self.stop_thread()  # 停止接收线程
        self.socket.close()
        self.context.term()

# on host machine
def test_pub(host_ip, host_port):
    zmq_ = ZmqWrapper(zmq.PUB, f"{host_ip}:{host_port}")
    arm = pb.Arm()
    # set arm
    zmq_.send(arm.SerializeToString())

# on slave machine
def test_sub(host_ip, host_port):
    def parse_zmq(zmq_msg):
        arm = pb.Arm()
        arm.ParseFromString(zmq_msg)
        print('stamp', arm.header.stamp.sec + arm.header.stamp.nanosec*1e-9)
        print('pose', arm.cartesian)
        print('joint_angles', arm.joint.position)
    zmq_sub = ZmqWrapper(zmq.SUB, f"{host_ip}:{host_port}")
    zmq_sub.start_receiver_thread(parse_zmq)


# on slave machine
def test_request(host_ip, host_port):
    zmq_req = ZmqWrapper(zmq.REQ, f"{host_ip}:{host_port}")
    request = pb.Request()
    request.key = "f2"
    response = pb.Response()
    response.ParseFromString(zmq_req.send(request.SerializeToString()))
    print(response.success, response.message)

# on host machine
def test_response(host_ip, host_port):
    zmq_response = ZmqWrapper(zmq.REP, f"{host_ip}:{host_port}")
    def response_service(msg, response):
        request = pb.Request()
        request.ParseFromString(msg)
        if request.key == "test":
            response.success = True
            response.message = "Reset success"
        # zmq_wrapper will take care of send response to slave
    zmq_response.start_receiver_thread(response_service)

# on host machine
def test_push(host_ip, host_port):
    zmq_ = ZmqWrapper(zmq.PUSH, f"{host_ip}:{host_port}")
    arm = pb.Arm()
    # set arm
    zmq_.send(arm.SerializeToString())

# on slave machine
def test_pull(host_ip, host_port):
    def parse_zmq(zmq_msg):
        arm = pb.Arm()
        arm.ParseFromString(zmq_msg)
        print('stamp', arm.header.stamp.sec + arm.header.stamp.nanosec*1e-9)
        print('pose', arm.cartesian)
        print('joint_angles', arm.joint.position)
    zmq_pull = ZmqWrapper(zmq.PULL, f"{host_ip}:{host_port}")
    zmq_pull.start_receiver_thread(parse_zmq)


def test_sub_action_bundle(host_ip="tcp://127.0.0.1", host_port=2019):
    def parse_zmq(zmq_msg):
        ab = pb.HandActionBundle()
        ab.ParseFromString(zmq_msg)
        msg = ''
        for glove in ab.gloves:
            msg+=f'{glove.header.key}->{glove.header.frame_id};'
        if msg: print('get action bundle:', msg)
    zmq_sub = ZmqWrapper(zmq.SUB, f"{host_ip}:{host_port}")
    zmq_sub.start_receiver_thread(parse_zmq)
    while True:
        time.sleep(1)
if __name__ == "__main__":
    # test_request("tcp://192.168.1.66", 2051)
    test_sub_action_bundle()