#!/usr/bin/env python3
import os
import pickle
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs_py import point_cloud2 as pc2


class ONNXPolicy:
    def __init__(self, model_path: str, vec_normalize_path: str):
        # Create ONNX Runtime session (prefer CUDA if available)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            # Fallback to CPU if CUDA EP not available
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean, self.std = self._load_vec_normalize_stats(vec_normalize_path)

    def _load_vec_normalize_stats(self, path: str):
        """Load VecNormalize stats (obs_rms.mean and var)."""
        with open(path, "rb") as f:
            try:
                vec_norm = pickle.load(f)  # works if same Python version
            except TypeError:
                # cross-Python compatibility
                vec_norm = pickle.load(f, encoding="latin1")

        if hasattr(vec_norm, "obs_rms"):
            mean = vec_norm.obs_rms.mean.astype(np.float32)
            std = np.sqrt(vec_norm.obs_rms.var).astype(np.float32)  # var -> std
            return mean, std
        raise ValueError("VecNormalize object has no 'obs_rms' attribute.")

    def normalize_input(self, data: np.ndarray, clip_obs: float = 100.0, epsilon: float = 1e-8):
        # SB3-like normalization (std already sqrt(var))
        normalized = (data - self.mean) / np.sqrt(self.std**2 + epsilon)
        return np.clip(normalized, -clip_obs, clip_obs)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run ONNX model prediction and map with expanded vy/yaw range."""
        inp = np.expand_dims(self.normalize_input(obs), axis=0).astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        actions = np.array(outputs[0][0], dtype=np.float32)  # shape (4,)

        low, high = -1.5, 1.5

        # vx: keep original scaling (will be overridden to 3.0 in test anyway)
        vx = 2.0 * ((actions[0] - 0.0) / (high - 0.0)) - 1.0

        # vy: expand range to [-6.0, 6.0] for stronger lateral avoidance
        vy_normalized = 2.0 * ((actions[1] - low) / (high - low)) - 1.0  # [-1, 1]
        vy = vy_normalized * 6.0  # [-6.0, 6.0]

        # vz: keep original
        vz = 2.0 * ((actions[2] - low) / (high - low)) - 1.0

        # vyaw: expand range to [-3.0, 3.0] for stronger turning
        vyaw_normalized = 2.0 * ((actions[3] - low) / (high - low)) - 1.0  # [-1, 1]
        vyaw = vyaw_normalized * 3.0  # [-3.0, 3.0]

        return np.array([vx, vy, vz, vyaw], dtype=np.float32)


class OnnxControllerNode(Node):
    def __init__(self):
        super().__init__("onnx_controller_node")

        # ---- Parameters (declare + read) ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("vec_normalize_path", "")
        self.declare_parameter("pointcloud_topic", "/pointcloud_features")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("expected_points", 256)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        vec_path = self.get_parameter("vec_normalize_path").get_parameter_value().string_value
        self.pc_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.expected_points = int(self.get_parameter("expected_points").value)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not os.path.isfile(vec_path):
            raise FileNotFoundError(f"VecNormalize pkl not found: {vec_path}")

        self.policy = ONNXPolicy(model_path, vec_path)

        # ---- Publisher / Subscriber ----
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.pc_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,  # best-effort sensor QoS
        )
        self.rand_point_sub = self.create_subscription(
            Bool,
            "/ca_rand_point_flag",
            self.rand_point_callback,
            qos_profile_sensor_data,  # best-effort sensor QoS
        )


        self.get_logger().info(
            f"ONNX controller ready. Subscribed to '{self.pc_topic}', publishing Twist to '{self.cmd_topic}'."
        )

    def pointcloud_callback(self, msg: PointCloud2):
        # Determine available fields; ensure we produce (N,4) with intensity fallback
        field_names = [f.name for f in msg.fields]
        use_intensity = "intensity" in field_names

        pts = []
        if use_intensity:
            for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                pts.append([p[0], p[1], p[2], p[3]])
        else:
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append([p[0], p[1], p[2], 0.0])

        points_array = np.asarray(pts, dtype=np.float32)

        # Pad/trim to (expected_points, 4)
        n = min(self.expected_points, len(points_array))
        input_data = np.zeros((self.expected_points, 4), dtype=np.float32)
        if n > 0:
            input_data[:n, :] = points_array[:n, :]

        # Predict and publish Twist
        actions = self.policy.predict(input_data)

        cmd = Twist()
        cmd.linear.x = float(actions[0])
        cmd.linear.y = float(actions[1])
        cmd.linear.z = float(actions[2])
        cmd.angular.z = float(actions[3])


        # 로깅: 위험 장애물 위치 정보 포함
        if not hasattr(self, '_cmd_log_initialized'):
            self._cmd_log_initialized = True
            import os
            log_dir = "/home/user/workspace/ros2/logs"
            os.makedirs(log_dir, exist_ok=True)
            with open("/home/user/workspace/ros2/logs/cmd.csv", "w") as f:
                f.write("vx,vy_final,vz,yaw_final,vy_mag,yaw_mag,closest_x,closest_y,avg_y,obstacle_side\n")


        with open("/home/user/workspace/ros2/logs/cmd.csv", "a") as f:
            f.write(f"{cmd.linear.x:.4f},{cmd.linear.y:.4f},{cmd.linear.z:.4f},{cmd.angular.z:.4f}\n")
        self.cmd_pub.publish(cmd)
    def rand_point_callback(self, msg: Bool):
        self.rand_point_flag = msg


def main(args=None):
    rclpy.init(args=args)
    node = OnnxControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
