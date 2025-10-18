## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os
import torchvision
import torch
from tqdm import tqdm
import math
import numpy as np
import cv2
from skimage import exposure
from decalib.utils.config import cfg as deca_cfg

run_this_file = 1

if run_this_file == 1:
    # from decalib.utils.renderer import SRenderY, set_rasterizer
    from decalib.datasets import datasets 
    # from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from decalib.deca import DECA

    image_size = 1024
    topology_path = '/content/drive/MyDrive/Towards-Realistic-Generative-3D-Face-Models/data/head_template.obj'
    uv_size = 1024
    rasterizer_type = 'pytorch3d'
    device = 'cuda'
    savefolder = '/content/drive/MyDrive/Towards-Realistic-Generative-3D-Face-Models/inference_test/out_data/uv_tex/'
    inputpath = '/content/drive/MyDrive/Towards-Realistic-Generative-3D-Face-Models/inference_test/in_data/'
    iscrop = True
    detector = 'fan'
    sample_step = 1
    useTex = True
    extractTex = True

    os.makedirs(savefolder, exist_ok=True)

    # Load test images
    testdata = datasets.TestData(inputpath, iscrop=iscrop, face_detector=detector, sample_step=sample_step, crop_size=1024)

    # Initialize DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config=deca_cfg, device=device)

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_normal(p1, p2, p3):
    return np.cross(p2-p1, p3-p1)

def mesh_angle(vertices, vertex_ids):
    normal = get_normal(np.array(vertices[vertex_ids[0]]), 
                        np.array(vertices[vertex_ids[1]]), 
                        np.array(vertices[vertex_ids[2]]))
    ang = int(angle(normal, [1,0,1])*360/math.pi)
    return ang

def remove_specular_highlights(uv_np):
    gray = cv2.cvtColor(uv_np, cv2.COLOR_BGR2GRAY)
    # Detect very bright pixels (specular highlights)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    # Smooth the bright regions
    uv_np_clean = cv2.inpaint(uv_np, mask, 5, cv2.INPAINT_TELEA)
    return uv_np_clean

def tex_correction(uv_texture, angle, gradient_ratio=0.25, center_band=150, blend_radius=40):
    """
    Combined UV Texture correction (without mean_texture or brightness normalization)
    -----------------------------------------------------------------
    Returns:
        uv_orig     : original texture (numpy)
        uv_out    : corrected texture (torch tensor)
    """

    device = uv_texture.device
    uv_orig = (uv_texture.detach().cpu().numpy() * 255).astype(np.uint8)
    uv_orig = remove_specular_highlights(uv_orig)
    uv_np = uv_orig
    h, w, _ = uv_np.shape
    mid_x = w // 2

    # ---------------------- (1) Determine the healthy side ----------------------
    if angle < 0:
        healthy_side, mask_side = "right", "left"
    else:
        healthy_side, mask_side = "left", "right"

    # ---------------------- (2) Histogram matching ----------------------
    left_half, right_half = uv_np[:, :mid_x], uv_np[:, mid_x:]
    if healthy_side == "left":
        right_half = exposure.match_histograms(right_half, left_half, channel_axis=-1)
    else:
        left_half = exposure.match_histograms(left_half, right_half, channel_axis=-1)
    uv_np = np.concatenate([left_half, right_half], axis=1).astype(np.uint8)

    # ---------------------- (3) Create blending mask ----------------------
    if mask_side == "right":
        diff = cv2.absdiff(cv2.flip(uv_np[:, :mid_x], 1), uv_np[:, mid_x:])
    else:
        diff = cv2.absdiff(cv2.flip(uv_np[:, mid_x:], 1), uv_np[:, :mid_x])

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask_half = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask_half = mask_half.astype(np.float32) / 255.0

    full_mask = np.zeros((h, w), dtype=np.float32)
    gradient_width = int(w * gradient_ratio)

    if mask_side == "right":
        full_mask[:, mid_x:] = mask_half
        grad = np.linspace(0, 1, gradient_width)[None, :]
        full_mask[:, mid_x - gradient_width:mid_x] = mask_half[:, :gradient_width] * grad
    else:
        full_mask[:, :mid_x] = mask_half
        grad = np.linspace(1, 0, gradient_width)[None, :]
        full_mask[:, mid_x:mid_x + gradient_width] = mask_half[:, -gradient_width:] * grad

    # Ensure central region continuity
    full_mask[:, mid_x - center_band:mid_x + center_band] = 1.0
    blurred_mask = cv2.GaussianBlur(full_mask, (35, 35), 0)

    # ---------------------- (4) Create mirrored reference ----------------------
    reference = uv_np.copy()
    if healthy_side == "left":
        reference[:, mid_x:] = cv2.flip(uv_np[:, :mid_x], 1)
    else:
        reference[:, :mid_x] = cv2.flip(uv_np[:, mid_x:], 1)

    # ---------------------- (5) Adaptive blending ----------------------
    blended = reference.astype(np.float32) * blurred_mask[..., None] + \
              uv_np.astype(np.float32) * (1 - blurred_mask[..., None])
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ---------------------- (6) Repair high difference areas ----------------------
    diff_check = cv2.cvtColor(cv2.absdiff(blended, uv_np), cv2.COLOR_BGR2GRAY)
    if np.mean(diff_check[:, mid_x - center_band:mid_x + center_band]) > 50:
        mask_inp = (full_mask > 0.5).astype(np.uint8)
        blended = cv2.inpaint(blended, mask_inp, 3, cv2.INPAINT_NS)

    uv_out = torch.from_numpy(blended.astype(np.float32) / 255.0).to(device)
    return uv_out, uv_orig

def tex_correction_eye(uv_texture, angle):
    if angle < 0:
        max_pixel = 512
        eye = 1
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[:,:max_pixel,:] = torch.flip(uv_texture, (1,))[:,:max_pixel,:]
        uv_texture[:200,:200,:] = eye
    else:
        max_pixel = -512
        eye = uv_texture[:200,-200:,:].clone()
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[:,max_pixel:,:] = torch.flip(uv_texture, (1,))[:,max_pixel:,:]
        uv_texture[:200,-200:,:] = eye
    return uv_texture

def tex_merge(uv_texture_r, uv_texture_c, uv_texture_l):
    max_pixel = 512
    arr = np.linspace(0, 1, max_pixel)          
    arr_flip = 1 - arr                         
    uv_texture_c[200:,:max_pixel,:] = (
        uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] +
        uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    )

    max_pixel = -512
    arr = np.linspace(0, 1, abs(max_pixel))     
    arr_flip = 1 - arr                         
    uv_texture_c[200:,max_pixel:,:] = (
        uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] +
        uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    )

    return uv_texture_c

def get_tex_from_img(images, get_cropped_img, deca):
    textures = torch.zeros_like(images).to('cuda')
    count=0

    for img in images:
        data_list = get_cropped_img.__getitem__(img*255)
        img_cropped = data_list['image'].to('cuda')[None,...]

        with torch.no_grad():
            codedict = deca.encode(torchvision.transforms.Resize(224)(img_cropped))
            codedict['images'] = img_cropped
            uv_tex, vertices, uv_face_eye_mask, uv_texture = deca.decode_tex(codedict)

            angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
            angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])
            avg_ang = int((angle1+angle2)/2)
            avg_ang = 90-(360-avg_ang)

            corrected_tex, orig_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)
 
            correct_tex = corrected_tex.permute(2,0,1)[None,...].to('cuda')
            correct_tex = correct_tex[:,:3,:,:]*uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-uv_face_eye_mask))
            textures[count] = correct_tex
            count+=1

    return textures

def main():
    # ========== PRECHECK ==========
    try:
        files = os.listdir(inputpath)
        print(f"[INFO] files in inputpath: {files[:50]}")
    except Exception as e:
        print(f"[WARN] cannot list inputpath: {e}")
        files = []

    print(f"[INFO] testdata length: {len(testdata)}")
    os.makedirs(savefolder, exist_ok=True)

    if len(testdata) == 0:
        print("[ERROR] testdata is empty. Make sure inputpath contains images and that datasets.TestData can read them.")
        return

    for i in tqdm(range(len(testdata))):
        try:
            item = testdata[i]
            name = item.get('imagename', f"sample_{i}")
            images = item.get('image', None)
            if images is None:
                print(f"[WARN] item {i} has no 'image' key, skipping.")
                continue

            images = images.to(device)[None, ...]
            with torch.no_grad():
                # Encode and decode
                codedict = deca.encode(torchvision.transforms.Resize(224)(images))
                codedict['images'] = images

                # Get all outputs, ignore extra values
                uv_tex, vertices, *_ = deca.decode_tex(codedict)

                # Compute angle
                angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
                angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])
                avg_ang = int((angle1 + angle2) / 2)
                avg_ang = 90 - (360 - avg_ang)
                print(f"\n  {name}: 📐avg_angle = {avg_ang}°")

                # Correct UV texture
                corrected_tex, orig_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)

                # Create folder for the image
                base_folder = os.path.join(savefolder, name)
                os.makedirs(base_folder, exist_ok=True)

                # Save images in the folder
                cv2.imwrite(os.path.join(base_folder, name + "_orig.png"), cv2.cvtColor(orig_tex, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(base_folder, name + "_corrected.png"), cv2.cvtColor((corrected_tex.cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                print(f"✅SAVED {name} → original and corrected inside {base_folder}")

        except Exception as e:
            print(f"[ERROR] processing item {i} ({name}): {e}")
            continue

if run_this_file == 1:
    main()
