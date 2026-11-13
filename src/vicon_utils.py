"""VICON GLB 읽기 도구 (Unreal Export GLB → 본 월드 좌표).

- 외부 라이브러리 없이 numpy만 사용 (glTF 2.0 바이너리 GLB)
- 애니메이션 키 시간마다 본의 월드 위치를 계산 (Three.js getWorldPosition과 같은 계산)
- 좌표계는 glTF 표준 그대로: Y-up, 단위 m
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class GlbMotion:
    """GLB 한 개의 본 애니메이션 (월드 좌표)."""

    path: Path
    animation_name: str
    times: np.ndarray  # (F,) 초
    bone_names: list[str]  # (J,) 스킨 관절 순서
    positions: np.ndarray  # (F, J, 3) m, Y-up

    @property
    def frame_count(self) -> int:
        return len(self.times)

    def joint(self, name: str) -> np.ndarray:
        """(F, 3) 한 본의 시간별 위치."""
        return self.positions[:, self.bone_names.index(name)]


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, _version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"GLB 파일이 아님: {path.name}")
    offset, gltf, binary = 12, None, b""
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:  # JSON
            gltf = json.loads(chunk)
        elif chunk_type == 0x004E4942:  # BIN
            binary = chunk
    if gltf is None:
        raise ValueError(f"GLB에 JSON 청크가 없음: {path.name}")
    return gltf, binary


def _accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
    size = _TYPE_SIZES[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0)
    count = acc["count"]
    item = size * dtype.itemsize
    if stride and stride != item:
        rows = [
            np.frombuffer(binary, dtype, count=size, offset=start + k * stride)
            for k in range(count)
        ]
        arr = np.stack(rows)
    else:
        arr = np.frombuffer(binary, dtype, count=size * count, offset=start).reshape(
            count, size
        )
    return arr.astype(np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(F, 4) xyzw → (F, 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def load_glb_motion(path: str | Path) -> GlbMotion:
    """GLB를 읽어 스킨 관절의 프레임별 월드 위치를 계산 (키 시간 그대로, 리샘플 없음)."""
    path = Path(path)
    gltf, binary = _read_glb(path)
    nodes = gltf["nodes"]
    if not gltf.get("animations"):
        raise ValueError(f"애니메이션이 없음: {path.name}")
    if not gltf.get("skins"):
        raise ValueError(f"스킨(본 목록)이 없음: {path.name}")

    anim = gltf["animations"][0]
    times = None
    tracks: dict[str, dict[int, np.ndarray]] = {
        "translation": {},
        "rotation": {},
        "scale": {},
    }
    for channel in anim["channels"]:
        sampler = anim["samplers"][channel["sampler"]]
        if sampler.get("interpolation", "LINEAR") != "LINEAR":
            raise ValueError(f"LINEAR 보간만 지원: {path.name}")
        t = _accessor(gltf, binary, sampler["input"])[:, 0]
        if times is None:
            times = t
        elif len(t) != len(times) or not np.allclose(t, times, atol=1e-5):
            raise ValueError(f"채널마다 키 시간이 다름: {path.name}")
        target = channel["target"]
        tracks[target["path"]][target["node"]] = _accessor(
            gltf, binary, sampler["output"]
        )

    frames = len(times)
    parent = [-1] * len(nodes)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i

    world_r: list[np.ndarray | None] = [None] * len(nodes)
    world_t: list[np.ndarray | None] = [None] * len(nodes)

    def compute(i: int) -> None:
        if world_r[i] is not None:
            return
        node = nodes[i]
        t = tracks["translation"].get(
            i, np.tile(node.get("translation", [0.0, 0.0, 0.0]), (frames, 1))
        )
        q = tracks["rotation"].get(
            i, np.tile(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), (frames, 1))
        )
        s = tracks["scale"].get(
            i, np.tile(node.get("scale", [1.0, 1.0, 1.0]), (frames, 1))
        )
        local = _quat_to_matrix(q) * s[:, None, :]
        p = parent[i]
        if p < 0:
            world_r[i], world_t[i] = local, t
        else:
            compute(p)
            world_r[i] = world_r[p] @ local
            world_t[i] = (world_r[p] @ t[..., None])[..., 0] + world_t[p]

    joints = gltf["skins"][0]["joints"]
    for j in joints:
        compute(j)

    names = [nodes[j].get("name", f"node{j}").split(":")[-1] for j in joints]
    positions = np.stack([world_t[j] for j in joints], axis=1)
    return GlbMotion(
        path=path,
        animation_name=anim.get("name", ""),
        times=times,
        bone_names=names,
        positions=positions,
    )


def _horizontal_deg(v: np.ndarray) -> float:
    """수평 방향 각도. 0° = +X, +90° = −Z (첫 T-pose 기준 왼쪽)."""
    return float(np.degrees(np.arctan2(-v[2], v[0])))


def facing_deg(motion: GlbMotion, seconds: float = 0.5) -> float:
    """처음 seconds초 동안 골반 기준으로 몸이 향한 각도.

    앞 = 위(+Y) × (오른골반 − 왼골반). 01 문서 P-2 정면 정렬과 같은 기준.
    """
    n = _first_frames(motion, seconds)
    lr = (motion.joint("RightUpLeg") - motion.joint("LeftUpLeg"))[:n].mean(axis=0)
    return _horizontal_deg(np.cross([0.0, 1.0, 0.0], lr))


def toe_deg(motion: GlbMotion, seconds: float = 0.5) -> float:
    """처음 seconds초 동안 발끝(발목 → 발가락 뿌리)이 향한 각도. 좌우와 무관한 앞 방향."""
    n = _first_frames(motion, seconds)
    toe = (
        motion.joint("LeftToeBase")
        - motion.joint("LeftFoot")
        + motion.joint("RightToeBase")
        - motion.joint("RightFoot")
    )[:n].mean(axis=0)
    return _horizontal_deg(toe)


def _first_frames(motion: GlbMotion, seconds: float) -> int:
    dt = float(np.median(np.diff(motion.times))) if motion.frame_count > 1 else 1.0
    return max(1, min(motion.frame_count, round(seconds / dt)))


def angle_diff_deg(a: float, b: float) -> float:
    """두 각도 차이의 절댓값 (0~180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)