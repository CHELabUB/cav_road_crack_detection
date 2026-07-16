# OBU transmission node for stage 1 and stage 3.
# author: Haosong Xiao,
# native import
import ast
import csv
import json
import os
import time

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

# commsignia APIs
from pycmssdk import (
    create_cms_api,
    MacAddr,
    RadioTxParams,
    WsmpTxHdrInfo,
    WsmpSendData,
)

V2X_STACK_IP = "192.168.3.54"
OBU_SEND_PSID = 1002
OBU_RECV_PSID = 1001
# OBU_RESULT_PSID = 1003


def prepare_OBU_data(psid):
    return WsmpSendData(
        radio=RadioTxParams(
            interface_id=1,
            dest_address=MacAddr(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF),
        ),
        wsmp_hdr=WsmpTxHdrInfo(psid=psid),
    )


class OBUSentNode(Node):
    def __init__(self, api):
        super().__init__("obu_sent_node")

        self.api = api
        self.obu_send_psid = OBU_SEND_PSID
        self.obu_recv_psid = OBU_RECV_PSID
        self.ready_interval_sec = 0.2
        self.crack_lat = None
        self.crack_lon = None
        self.pending_ack_id = None
        self.process_finished = 0
        self.test_num = None

        # publisher
        self.gps_pub = self.create_publisher(Float64MultiArray, "/obu/gps", 10)

        # for the OBU/RSU communication
        self.api.wsmp_rx_subscribe(self.obu_recv_psid, self.rx_callback)
        self.get_logger().info("[OBU] Connected and subscribed")

        # callbacks
        self.create_timer(max(0.1, self.ready_interval_sec), self.ready_timer_callback)
        self.create_timer(0.1, self.gps_publish_callback)

        # subscriber
        self.test_num_sub = self.create_subscription(String, '/test_number', self.test_num_callback, 50)
        # self.detection_length_sub = self.create_subscription(String, '/detection_length', self.length_results_callback, 50)
        self.process_finished_sub = self.create_subscription(String, '/process_finished', self.process_finished_callback, 10)
        self.result_dir = None

    def extract_gps(self, msg):
        if not isinstance(msg, dict):
            return None
        self.get_logger().info(f"[OBU] Extracting GPS from message: {msg}")
        lat = msg.get("gps_lat")
        lon = msg.get("gps_lon")

        if lat is None or lon is None:
            return None

        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    def rx_callback(self, _if, _hdr, buffer):
        try:
            msg = json.loads(buffer.decode("utf-8"))
            self.get_logger().info(f"[OBU RX] {msg}")

            if msg.get("type") == "ack" and msg.get("ack_for") == "length_results":
                self.pending_ack_id = msg.get("id")
                return

            if msg.get("type") == "gps_request":
                request_id = msg.get("request_id")
                self.send_msg({
                    "type": "ack",
                    "ack_for": "gps_request",
                    "request_id": request_id,
                })

            gps = self.extract_gps(msg)
            if gps is not None:
                lat, lon = gps
                self.crack_lat = lat
                self.crack_lon = lon
                self.get_logger().info(f"[OBU GPS] lat={lat}, lon={lon}")
        except Exception:
            self.get_logger().warn(f"[OBU] Raw buffer: {buffer}")

    def test_num_callback(self, msg: String):
        self.test_num = msg.data
        # self.setup_test_folder()

    def gps_publish_callback(self):
        if self.crack_lat is not None and self.crack_lon is not None:
            gps_msg = Float64MultiArray()
            gps_msg.data = [self.crack_lat, self.crack_lon]
            self.gps_pub.publish(gps_msg)
            self.get_logger().warn(f"[OBU GPS Published] lat={self.crack_lat}, lon={self.crack_lon}")

    def send_msg(self, payload):
        try:
            raw = json.dumps(payload).encode("utf-8")
            self.api.wsmp_send(
                prepare_OBU_data(self.obu_send_psid),
                buffer=raw,
            )
            self.get_logger().info(f"[OBU TX] {payload}")
        except Exception as error:
            self.get_logger().error(f"[OBU TX FAILED] {repr(error)}")

    def ready_timer_callback(self):
        self.send_msg({"type": "ready"})

    def send_crack_log(self):
        try:
            with open(self.result_dir, newline="") as f:
                for row in csv.DictReader(f):
                    image_id = row["image_name"]
                    self.pending_ack_id = None
                    self.send_msg({
                        "type": "length_results",
                        "id": image_id,
                        "computed_timestamp": int(row["computed_timestamp"]),
                        "synced_crop_timestamp": int(row["synced_crop_timestamp"]),
                        "crack_length_m": float(row["crack_length_m"]),
                    })
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        if self.pending_ack_id == image_id:
                            self.get_logger().info(f"[OBU] ACK received for {image_id}")
                            break
                        time.sleep(0.01)
                    else:
                        self.get_logger().warn(f"[OBU] No ACK for {image_id}, continuing")
        except Exception as e:
            self.get_logger().error(f"[OBU] Failed to send crack log: {repr(e)}")

    def process_finished_callback(self, msg: String):
        self.process_finished = int(msg.data)
        self.get_logger().info(f"[OBU] process_finished={self.process_finished}")
        if self.process_finished == 1:
            if self.test_num is not None:
                _node_dir = os.path.dirname(os.path.abspath(__file__))
                self.result_dir = os.path.join(_node_dir, '..', 'data', f"Test_{self.test_num}", "crack_length_log.txt")
                self.send_crack_log()

    # def length_results_callback(self, msg: String):
    #     results = ast.literal_eval(msg.data)
    #     self.get_logger().info(f"[OBU] Received length results: {results}")
    #     # self.send_msg({"type": "length_results", "results": results})


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        with create_cms_api(host=V2X_STACK_IP) as api:
            node = OBUSentNode(api)
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
