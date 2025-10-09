# video.py
import os
import cv2
import torch
import numpy as np
from face_view_renderer import (
    render_rgb,
    load_mesh_center_radius,
    find_frontal_angles,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMG_SIZE = 1024

# ---------- Video generator ----------
def save_rotation_video(mesh_path, out_video, n_frames=80, fps=15, delta_yaw=90.0):
    mesh, center, radius = load_mesh_center_radius(mesh_path)
    yaw, roll = find_frontal_angles(mesh, center, radius)
    print(f"Best yaw: {yaw:.2f}, roll: {roll:.2f}")

    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    forward = np.linspace(yaw - delta_yaw, yaw + delta_yaw, n_frames)
    backward = np.linspace(yaw + delta_yaw, yaw - delta_yaw, n_frames)[1:-1]
    angles = np.concatenate([forward, backward])

    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMG_SIZE, IMG_SIZE))

    for i, a in enumerate(angles):
        img = render_rgb(mesh, center, radius, a, roll_deg=roll, out_size=IMG_SIZE)
        out.write(img)
        if i % 10 == 0:
            print(f"Rendering frame({i})...")

    out.release()
    print(f"✅Video saved at: {out_video}")

# ---------- Main ----------
if __name__ == "__main__":
    in_path = "inference_test/out_data/"
    out_path = "inference_test/out_data/videos/"
    os.makedirs(out_path, exist_ok=True)

    subdirs = sorted([d for d in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, d))])

    for name in subdirs:
        mesh_path = os.path.join(in_path, name, f"{name}.obj")
        if not os.path.exists(mesh_path):
            continue
        out_file = os.path.join(out_path, f"{name}.mp4")
        print(f"🎥Processing: {name}")
        save_rotation_video(mesh_path, out_file)
