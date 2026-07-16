# Code written on RSU side to communicate with OBU node in the pipeline, written with help from copilot. 
# native imports
import argparse
import json
import os
import time
# commsignia imports
from pycmssdk import (
    create_cms_api,
    MacAddr,
    RadioTxParams,
    WsmpTxHdrInfo,
    WsmpSendData,
)

# Example usage: python3 detection_pipeline/RSU_code/RSU_receive.py 1 -> change 1 to the designated ID that we are trying to test.
V2X_STACK_IP = "192.168.0.54"
RSU_SEND_PSID = 1001   # RSU -> OBU
RSU_RECV_PSID = 1002   # OBU -> RSU
File_time = int(time.time() * 1000)

# load crack location
Cracks_json_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cracks.json")


def make_send_data(psid):
    return WsmpSendData(
        radio=RadioTxParams(
            interface_id=1,
            dest_address=MacAddr(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF),
        ),
        wsmp_hdr=WsmpTxHdrInfo(psid=psid),
    )


class RSUNode:
    def __init__(self, api, crack):
        self.api = api
        self.last_msg = None
        self.result_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsu_result_log.txt")
        self.crack_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rsu_crack_log_{File_time}.txt")
        self.crack = crack

    def log_length_result(self, msg):
        needs_header = not os.path.exists(self.crack_log_path) or os.path.getsize(self.crack_log_path) == 0
        with open(self.crack_log_path, "a") as f:
            if needs_header:
                f.write("image_name,computed_timestamp,synced_crop_timestamp,crack_length_m,received_time_ms\n")
            f.write(
                f"{msg.get('id')},"
                f"{msg.get('computed_timestamp')},"
                f"{msg.get('synced_crop_timestamp')},"
                f"{msg.get('crack_length_m')},"
                f"{int(time.time() * 1000)}\n"
            )

    def rx_callback(self, _if, _hdr, buffer):
        try:
            decoded = buffer.decode("utf-8")
            msg = json.loads(decoded)
            self.last_msg = msg
            print("[RSU RX]", msg)

            if msg.get("type") == "length_results":
                self.log_length_result(msg)
                self.send_msg({
                    "type": "ack",
                    "ack_for": "length_results",
                    "id": msg.get("id"),
                })
        except Exception as e:
            print("[RSU] Decode failed:", repr(e))
            print("[RSU] Raw buffer:", buffer)

    def send_msg(self, payload):
        try:
            self.api.wsmp_send(
                make_send_data(RSU_SEND_PSID),
                buffer=json.dumps(payload).encode("utf-8"),
            )
            print("[RSU TX]", payload)
            return True
        except Exception as e:
            print("[RSU] Send failed:", repr(e))
            return False

    def wait_for(self, target_type, request_id=None, timeout=2.0):
        start = time.time()

        while time.time() - start < timeout:
            if self.last_msg is not None:
                msg = self.last_msg
                self.last_msg = None

                print("[RSU DEBUG] got:", msg)

                if msg.get("type") != target_type:
                    print("[RSU DEBUG] ignored type:", msg.get("type"))
                    continue

                if request_id is not None and msg.get("request_id") != request_id:
                    print("[RSU DEBUG] ignored request_id:", msg.get("request_id"))
                    continue

                return msg

            time.sleep(0.01)

        return None

    def run(self):
        print("[RSU] Subscribing PSID:", RSU_RECV_PSID)
        self.api.wsmp_rx_subscribe(RSU_RECV_PSID, self.rx_callback)
        print("[RSU] Subscribe done")
        print("[RSU] Running GPS request protocol...")

        cycle = 0

        while True:
            print('clcye:', cycle)
            print("\n[RSU] Waiting for OBU ready...")
            ready_msg = self.wait_for("ready", timeout=20.0)

            if ready_msg is None:
                print("[RSU] Timeout waiting for ready")
                continue

            print("[RSU] Ready received:", ready_msg)

            cycle += 1
            request_id = f"req_{cycle}"

            gps_request = {
                "type": "gps_request",
                "request_id": request_id,
                "crack_id": self.crack["id"],
                "gps_lat": self.crack["lat"],
                "gps_lon": self.crack["lon"],
                "send_time_ms": int(time.time() * 1000),
            }

            if not self.send_msg(gps_request):
                continue

            print("[RSU] Waiting for gps_request ACK...")
            ack_msg = self.wait_for("ack", request_id=request_id, timeout=10.0)

            # check this, I think this may be redundant
            if ack_msg is None:
                print("[RSU] Timeout waiting for gps_request ACK")
                continue

            if ack_msg.get("ack_for") != "gps_request":
                print("[RSU] Invalid ACK:", ack_msg)
                continue

            print("[RSU] gps_request ACK received")

            result_msg = self.wait_for("length_results", timeout=30.0)
            print("[RSU] start asking results")
            if result_msg is None:
                print("[RSU] Timeout waiting for result")
                continue

            print("[RSU] Result received:", result_msg)
            print(
                f"[RSU] Parsed result id={result_msg.get('id')}, "
                f"crack_length_m={result_msg.get('crack_length_m')}"
            )

            result_request = {
                "type": "ack",
                "ack_for": "length_results",
                "id": result_msg.get('id'),
            }
            self.send_msg(result_request)
            print("[RSU] Cycle complete:", request_id)
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="RSU receiver — specify which crack to target")
    parser.add_argument("crack_id", type=int, help="Crack ID from cracks.json")
    args = parser.parse_args()

    with open(Cracks_json_PATH, "r") as f:
        cracks = json.load(f)
    crack = next((c for c in cracks if c["id"] == args.crack_id), None)
    if crack is None:
        available = [c["id"] for c in cracks]
        parser.error(f"Crack ID {args.crack_id} not found in cracks.json. Available IDs: {available}")

    print(f"[RSU] Targeting crack {crack['id']} at ({crack['lat']}, {crack['lon']})")

    with create_cms_api(host=V2X_STACK_IP) as api:
        node = RSUNode(api, crack)
        node.run()


if __name__ == "__main__":
    main()