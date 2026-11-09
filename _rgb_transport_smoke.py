"""Synthetic codec/client smoke only: no physical camera or robot is opened."""
import base64
import threading
import time

import numpy as np
import zmq

from d455_rgb import _encode_rgb_png
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

rgb = np.zeros((16, 16, 3), dtype=np.uint8)
rgb[:, :8] = [231, 17, 63]
rgb[:, 8:] = [9, 151, 247]
context = zmq.Context()
ready = threading.Event()
stop = threading.Event()
port = []

def publish():
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    port.append(socket.bind_to_random_port('tcp://127.0.0.1'))
    ready.set()
    payload = base64.b64encode(_encode_rgb_png(rgb, transport=True)).decode('ascii')
    try:
        while not stop.wait(0.02):
            socket.send_json({'images': {'top': payload}, 'timestamps': {'top': time.time()}})
    finally:
        socket.close()

producer = threading.Thread(target=publish)
producer.start()
assert ready.wait(5)
camera = ZMQCamera(ZMQCameraConfig(server_address='127.0.0.1', port=port[0], camera_name='top', width=16, height=16, fps=30, color_mode='rgb', timeout_ms=200, warmup_s=0))
try:
    camera.connect(warmup=False)
    received = camera.async_read(timeout_ms=2000)
    np.testing.assert_array_equal(received, rgb)
    print('PASS: synthetic RGB patches survive PNG + actual LeRobot ZMQCamera byte-for-byte; no hardware opened')
finally:
    if camera.is_connected:
        camera.disconnect()
    stop.set()
    producer.join(2)
    context.term()
