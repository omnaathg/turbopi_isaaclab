"""Run an ACT + language + CVAE checkpoint on the mountain cliff scene."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Drive TurboPi with a mountain ACT language checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", choices=("go_left", "go_right"), default="go_left")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", choices=("overview", "chase", "robot", "isometric"), default="chase")
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--control_mode", choices=("dynamic", "kinematic"), default="dynamic")
parser.add_argument(
    "--controller",
    choices=("policy", "route_follower"),
    default="policy",
    help="Use the ACT policy or a deterministic route follower for reliable lecture/demo renders.",
)
parser.add_argument("--policy_device", default="auto")
parser.add_argument("--vx_cap", type=float, default=0.45)
parser.add_argument("--vy_cap", type=float, default=0.35)
parser.add_argument("--wz_cap", type=float, default=2.0)
parser.add_argument("--smoothing", type=float, default=0.35)
parser.add_argument("--settle_steps", type=int, default=24)
parser.add_argument("--camera_warmup_steps", type=int, default=12)
parser.add_argument("--route_target_speed", type=float, default=0.20)
parser.add_argument("--route_min_speed", type=float, default=0.05)
parser.add_argument("--route_lookahead", type=float, default=0.18)
parser.add_argument("--route_switch_distance", type=float, default=0.10)
parser.add_argument("--route_heading_gain", type=float, default=2.0)
parser.add_argument("--route_lookahead_gain", type=float, default=0.8)
parser.add_argument("--no_rollers", action="store_true")
parser.add_argument("--save_video", type=str, default=None)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_output_dir", type=str, default=None, help="Optional directory for multi-view inference MP4s.")
parser.add_argument("--video_width", type=int, default=1920)
parser.add_argument("--video_height", type=int, default=1080)
parser.add_argument(
    "--video_views",
    type=str,
    default="robot,chase,isometric",
    help="Comma-separated video views to record. Choices: robot,chase,isometric.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from act_policy.runtime import ACTPolicyRuntime, ACTRuntimeConfig
from common import (
    CAMERA_LINK_TO_SENSOR_POS,
    CAMERA_LINK_TO_SENSOR_ROT,
    PERSPECTIVE_CAMERA_PATH,
    ROBOT_CAMERA_PATH,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot_pose,
    resolve_asset_usd,
    set_robot_camera_mount,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from mountain_cliff_scene import MountainCliffSceneCfg, design_mountain_cliff_scene, route_waypoints, start_pose

CAMERA_POS = (0.140, 0.0, 0.115)
CAMERA_ROT = (0.987688, 0.0, -0.156434, 0.0)
POLICY_CAMERA_PATH = "/World/TurboPiPolicyRobotCamera"
VIDEO_CAMERA_ROOT = "/World/TurboPiInferenceVideoCamera"
VIDEO_VIEWS = ("robot", "chase", "isometric")


class StopFlag:
    requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[drive] signal {signum}; stopping.", flush=True)


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    return x, y, wrap_to_pi(float(yaw_t[0].item()))


def build_camera(
    width: int,
    height: int,
    *,
    prim_path: str = POLICY_CAMERA_PATH,
    focal_length: float = 18.0,
) -> Camera:
    return Camera(
        CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 100.0),
            ),
        )
    )


def build_policy_camera(width: int, height: int) -> Camera:
    return Camera(
        CameraCfg(
            prim_path=ROBOT_CAMERA_PATH,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(
                pos=tuple(CAMERA_LINK_TO_SENSOR_POS),
                rot=tuple(CAMERA_LINK_TO_SENSOR_ROT),
                convention="opengl",
            ),
        )
    )


def parse_video_views(views_arg: str) -> tuple[str, ...]:
    views = tuple(view.strip() for view in views_arg.split(",") if view.strip())
    unknown = tuple(view for view in views if view not in VIDEO_VIEWS)
    if unknown:
        raise ValueError(f"Unknown video view(s): {', '.join(unknown)}. Valid views: {', '.join(VIDEO_VIEWS)}")
    if not views:
        raise ValueError("At least one video view must be selected.")
    return views


def build_video_cameras(width: int, height: int, views: tuple[str, ...]) -> dict[str, Camera]:
    return {
        view: build_camera(
            width,
            height,
            prim_path=f"{VIDEO_CAMERA_ROOT}_{view}",
            focal_length=20.0,
        )
        for view in views
    }


def update_policy_camera(camera: Camera, robot) -> None:
    base_pos = robot.data.root_pos_w[0]
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = float(yaw_t[0].item())
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    def to_world(offset: tuple[float, float, float]) -> list[float]:
        x, y, z = offset
        return [
            float(base_pos[0].item()) + cos_yaw * x - sin_yaw * y,
            float(base_pos[1].item()) + sin_yaw * x + cos_yaw * y,
            float(base_pos[2].item()) + z,
        ]

    eye = to_world((0.18, 0.0, 0.18))
    target = to_world((1.35, 0.0, 0.04))
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=robot.device),
        torch.tensor([target], dtype=torch.float32, device=robot.device),
    )


def camera_pose_from_robot(robot, eye_offset: tuple[float, float, float], target_offset: tuple[float, float, float]) -> tuple[list[float], list[float]]:
    base_pos = robot.data.root_pos_w[0]
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = float(yaw_t[0].item())
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    def to_world(offset: tuple[float, float, float]) -> list[float]:
        x, y, z = offset
        return [
            float(base_pos[0].item()) + cos_yaw * x - sin_yaw * y,
            float(base_pos[1].item()) + sin_yaw * x + cos_yaw * y,
            float(base_pos[2].item()) + z,
        ]

    return to_world(eye_offset), to_world(target_offset)


def set_camera_pose(camera: Camera, eye: list[float], target: list[float], device: str) -> None:
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=device),
        torch.tensor([target], dtype=torch.float32, device=device),
    )


def isometric_pose(scene_cfg: MountainCliffSceneCfg) -> tuple[list[float], list[float]]:
    return [3.10, -3.30, scene_cfg.road_z + 1.80], [0.35, 1.15, scene_cfg.road_z - 0.10]


def update_video_camera(camera: Camera, robot, scene_cfg: MountainCliffSceneCfg, view: str, dt: float) -> None:
    if view == "robot":
        eye, target = camera_pose_from_robot(robot, (0.18, 0.0, 0.18), (1.35, 0.0, 0.04))
    elif view == "chase":
        eye, target = camera_pose_from_robot(robot, (-1.65, -0.08, 0.72), (0.85, 0.02, 0.08))
    elif view == "isometric":
        eye, target = isometric_pose(scene_cfg)
    else:
        raise ValueError(f"Unknown video view: {view}")
    set_camera_pose(camera, eye, target, robot.device)
    camera.update(dt=dt)


def update_video_cameras(video_cameras: dict[str, Camera] | None, robot, scene_cfg: MountainCliffSceneCfg, dt: float) -> None:
    if not video_cameras:
        return
    for view, video_camera in video_cameras.items():
        update_video_camera(video_camera, robot, scene_cfg, view, dt)


def open_video_writers(task_name: str, views: tuple[str, ...]) -> dict[str, cv2.VideoWriter]:
    if not args_cli.video_output_dir:
        return {}
    video_dir = Path(args_cli.video_output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, cv2.VideoWriter] = {}
    for view in views:
        path = video_dir / f"mountain_act_inference_{task_name}_{view}_{args_cli.video_width}x{args_cli.video_height}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.video_fps),
            (args_cli.video_width, args_cli.video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        writers[view] = writer
        print(f"[drive-video] recording {view} -> {path}", flush=True)
    return writers


def _draw_minimap(
    frame: np.ndarray,
    pose: tuple[float, float, float],
    route: tuple[tuple[float, float], ...],
    task_name: str,
    size: int = 120,
    margin: int = 8,
) -> np.ndarray:
    """Draw a top-down minimap in the bottom-left corner of the frame."""
    all_x = [p[0] for p in route]
    all_y = [p[1] for p in route]
    pad = 0.3
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    def to_px(wx: float, wy: float) -> tuple[int, int]:
        px = int((wx - x_min) / max(x_max - x_min, 1e-6) * (size - 1))
        py = int((1.0 - (wy - y_min) / max(y_max - y_min, 1e-6)) * (size - 1))
        return px, py

    mini = np.zeros((size, size, 3), dtype=np.uint8)
    mini[:] = (30, 30, 30)
    color = (60, 160, 255) if task_name == "go_left" else (50, 180, 80)
    pts = np.array([to_px(wx, wy) for wx, wy in route], dtype=np.int32)
    cv2.polylines(mini, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    for wx, wy in route:
        cv2.circle(mini, to_px(wx, wy), 3, (220, 220, 100), -1, lineType=cv2.LINE_AA)
    rx, ry, _ = pose
    rpx, rpy = to_px(rx, ry)
    cv2.circle(mini, (rpx, rpy), 5, (255, 80, 80), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(mini, (0, 0), (size - 1, size - 1), (120, 120, 120), 1)

    h, w = frame.shape[:2]
    y0 = h - size - margin
    x0 = margin
    frame = frame.copy()
    frame[y0:y0 + size, x0:x0 + size] = mini
    return frame


def write_video_frames(
    video_cameras: dict[str, Camera] | None,
    video_writers: dict[str, cv2.VideoWriter],
    robot,
    scene_cfg: MountainCliffSceneCfg,
    dt: float,
    pose: tuple[float, float, float] | None = None,
    route: tuple[tuple[float, float], ...] | None = None,
    task_name: str = "",
) -> None:
    if not video_cameras or not video_writers:
        return
    update_video_cameras(video_cameras, robot, scene_cfg, dt)
    for view, writer in video_writers.items():
        frame = cv2.cvtColor(rgb_frame(video_cameras[view]), cv2.COLOR_RGB2BGR)
        if pose is not None and route is not None:
            frame = _draw_minimap(frame, pose, route, task_name)
        writer.write(frame)


def close_video_writers(video_writers: dict[str, cv2.VideoWriter]) -> None:
    for writer in video_writers.values():
        writer.release()


def rgb_frame(camera: Camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def integrate(pose, command, dt):
    x, y, yaw = pose
    vx, vy, wz = command[:3]
    yaw_mid = yaw + 0.5 * wz * dt
    return (
        x + (vx * np.cos(yaw_mid) - vy * np.sin(yaw_mid)) * dt,
        y + (vx * np.sin(yaw_mid) + vy * np.cos(yaw_mid)) * dt,
        wrap_to_pi(yaw + wz * dt),
    )


def write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, command, root_z):
    x, y, yaw = pose
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = x
    root_pose[:, 1] = y
    root_pose[:, 2] = root_z
    yaw_t = torch.full((robot.num_instances,), yaw, dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device))
    command_t = torch.as_tensor(np.asarray(command[:3], dtype=np.float32)[None, :], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def apply_dynamic(robot, wheel_joint_ids, arm_joint_ids, command):
    command_t = torch.as_tensor(np.asarray(command[:3], dtype=np.float32)[None, :], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def compute_route_command(pose: tuple[float, float, float], route: tuple[tuple[float, float], ...], segment_index: int) -> tuple[np.ndarray, int]:
    x, y, yaw = pose
    segment_index = min(segment_index, len(route) - 2)
    start = np.asarray(route[segment_index], dtype=np.float32)
    goal = np.asarray(route[segment_index + 1], dtype=np.float32)
    pos = np.asarray([x, y], dtype=np.float32)
    seg = goal - start
    length = float(np.linalg.norm(seg))
    if length <= 1e-6:
        return np.zeros(4, dtype=np.float32), segment_index
    dist_to_goal = float(np.linalg.norm(goal - pos))
    if dist_to_goal <= args_cli.route_switch_distance and segment_index < len(route) - 2:
        segment_index += 1
        start = np.asarray(route[segment_index], dtype=np.float32)
        goal = np.asarray(route[segment_index + 1], dtype=np.float32)
        seg = goal - start
        length = float(np.linalg.norm(seg))
    t = float(np.clip(np.dot(pos - start, seg) / max(length * length, 1e-9), 0.0, 1.0))
    lookahead_t = min(1.0, t + args_cli.route_lookahead / max(length, 1e-6))
    target = start + lookahead_t * seg
    delta = target - pos
    target_bx = np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1]
    target_by = -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]
    point_error = float(np.arctan2(target_by, max(float(target_bx), 0.04)))
    heading = float(np.arctan2(seg[1], seg[0]))
    yaw_error = wrap_to_pi(heading - yaw)
    speed_scale = float(np.clip(1.0 - abs(yaw_error) / 1.25, 0.20, 1.0))
    vx = max(args_cli.route_min_speed, args_cli.route_target_speed * speed_scale)
    if abs(yaw_error) > 1.15:
        vx = 0.0
    wz = float(np.clip(args_cli.route_heading_gain * yaw_error + args_cli.route_lookahead_gain * point_error, -args_cli.wz_cap, args_cli.wz_cap))
    return np.asarray([vx, 0.0, wz, 0.0], dtype=np.float32), segment_index


def main() -> None:
    runtime = ACTPolicyRuntime(
        args_cli.checkpoint,
        task=args_cli.task,
        device=args_cli.policy_device,
        runtime_cfg=ACTRuntimeConfig(vx_cap=args_cli.vx_cap, vy_cap=args_cli.vy_cap, wz_cap=args_cli.wz_cap, smoothing=args_cli.smoothing),
    )
    scene_cfg = MountainCliffSceneCfg()
    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    video_dt = 1.0 / max(args_cli.video_fps, 1e-6)
    substeps = max(1, int(round(control_dt / physics_dt)))
    video_substeps = max(1, int(round(video_dt / physics_dt)))
    render_interval = video_substeps if args_cli.video_output_dir else substeps
    if not args_cli.headless or bool(getattr(args_cli, "livestream", 0)):
        render_interval = 1
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device))
    design_mountain_cliff_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CAMERA_LINK_TO_SENSOR_POS, CAMERA_LINK_TO_SENSOR_ROT)
    camera = build_policy_camera(runtime.image_width, runtime.image_height)
    video_views = parse_video_views(args_cli.video_views)
    video_cameras = build_video_cameras(args_cli.video_width, args_cli.video_height, video_views) if args_cli.video_output_dir else None
    sim.reset()
    start_position, start_yaw = start_pose(scene_cfg)
    root_z = scene_cfg.road_z + scene_cfg.start_height
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    sim.play()
    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode("overview" if args_cli.view == "isometric" else args_cli.view, sim, robot, viewport)
    if args_cli.view == "isometric" and viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
        sim.set_camera_view(eye=[1.75, -2.50, scene_cfg.road_z + 1.35], target=[0.75, 0.95, scene_cfg.road_z])
        active_view = "isometric"
    elif args_cli.view == "robot" and viewport is not None:
        viewport.set_active_camera(POLICY_CAMERA_PATH)
        active_view = "robot"
    pose = (float(start_position[0]), float(start_position[1]), float(start_yaw))
    route = route_waypoints(scene_cfg, args_cli.task)
    segment_index = 0
    for _ in range(max(1, args_cli.settle_steps + args_cli.camera_warmup_steps)):
        sim.step()
        robot.update(physics_dt)
        camera.update(dt=physics_dt)
        update_video_cameras(video_cameras, robot, scene_cfg, physics_dt)

    writer = None
    if args_cli.save_video:
        path = Path(args_cli.save_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (runtime.image_width, runtime.image_height))
    video_writers = open_video_writers(args_cli.task, video_views)
    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)
    print(f"[drive] checkpoint={args_cli.checkpoint} task={args_cli.task} params={runtime.model.parameter_count():,}")
    elapsed = 0.0
    next_video_time = 0.0
    try:
        while simulation_app.is_running() and not stop_flag.requested:
            frame = rgb_frame(camera)
            if args_cli.controller == "route_follower":
                command, segment_index = compute_route_command(pose, route, segment_index)
            else:
                _raw, command = runtime.predict(frame)
            if writer is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            for substep_index in range(substeps):
                if args_cli.control_mode == "kinematic":
                    pose = integrate(pose, command, physics_dt)
                    write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, command, root_z)
                else:
                    apply_dynamic(robot, wheel_joint_ids, arm_joint_ids, command)
                sim.step()
                robot.update(physics_dt)
                if active_view == "chase":
                    update_chase_camera(robot, viewport)
                if args_cli.control_mode == "dynamic":
                    pose = get_pose(robot)
                sim_time = elapsed + (substep_index + 1) * physics_dt
                if (
                    video_writers
                    and sim_time + 0.5 * physics_dt >= next_video_time
                    and (args_cli.duration <= 0 or next_video_time <= args_cli.duration + 1e-9)
                ):
                    write_video_frames(video_cameras, video_writers, robot, scene_cfg, physics_dt,
                                       pose=pose, route=route, task_name=args_cli.task)
                    next_video_time += video_dt
            camera.update(dt=control_dt)
            elapsed += control_dt
            if args_cli.duration > 0 and elapsed >= args_cli.duration:
                break
    finally:
        if writer is not None:
            writer.release()
        close_video_writers(video_writers)


def close_app_and_exit(code: int = 0) -> None:
    timer = threading.Timer(5.0, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    try:
        simulation_app.close()
    finally:
        timer.cancel()
        os._exit(code)


if __name__ == "__main__":
    main()
    close_app_and_exit(0)
