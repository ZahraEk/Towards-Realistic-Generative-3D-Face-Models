## This file has been taken from DECA and modified ##

import torch
import os
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torchvision
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from PIL import Image


def main(args):
    savefolder = args.savefolder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(savefolder, exist_ok=True)
    args.use_mica = True

    # Load test images
    testdata = datasets.TestData(
        args.inputpath,
        iscrop=args.iscrop,
        face_detector=args.detector,
        sample_step=args.sample_step,
        crop_size=1024,
        use_mica=args.use_mica
    )

    # Run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device, use_mica=args.use_mica)

    # Save options
    args.saveDepth = False
    args.saveKpt = False
    args.saveObj = True
    args.saveMat = False
    args.saveVis = True
    args.saveImages = False
    args.render_orig = False

    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        arcface_inp = testdata[i].get('arcface_inp', None)

        with torch.no_grad():
            if arcface_inp is not None:
                arcface_inp = arcface_inp.to(device)[None, ...]

            codedict = deca.encode(torchvision.transforms.Resize(224)(images), arcface_inp)
            codedict['images'] = images
            opdict, visdict = deca.decode(codedict, name)

            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1, 2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(
                    codedict, name, render_orig=True,
                    original_image=original_image, tform=tform
                )
                orig_visdict['inputs'] = original_image

        # Create subfolder for each result
        if any([args.saveDepth, args.saveKpt, args.saveObj, args.saveMat, args.saveImages]):
            os.makedirs(os.path.join(savefolder ,name), exist_ok=True)

        # -- Save outputs
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            depth_img = util.tensor2image(depth_image[0], convert_to_bgr=False)
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), depth_img[:, :, ::-1])

        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'),
                       opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'),
                       opdict['landmarks3d'][0].cpu().numpy())

        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict, codedict)

        if args.saveMat:
            opdict_npy = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict_npy)

        if args.saveVis:
            vis_img = deca.visualize(visdict)
            cv2.imwrite(os.path.join(savefolder, name, name + '_vis.jpg'), vis_img[:, :, ::-1])

            if args.render_orig:
                orig_vis_img = deca.visualize(orig_visdict)
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), orig_vis_img[:, :, ::-1])
               
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images',
                             'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict:
                    continue

                # Save normal image
                image = util.tensor2image(visdict[vis_name][0], convert_to_bgr=False)
                Image.fromarray(image).save(os.path.join(savefolder, name, name + f'_{vis_name}.jpg'))

                # Save original-size rendered image 
                if args.render_orig:
                    orig_image = util.tensor2image(orig_visdict[vis_name][0], convert_to_bgr=False)
                    orig_image_bgr = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                    Image.fromarray(orig_image).save(os.path.join(savefolder, name, name, f'orig_{name}_{vis_name}.jpg'))

    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlbedoGAN')

    parser.add_argument('-i', '--inputpath', default='inference_test/in_data', type=str,
                        help='Path to the test data, can be image folder, image path, image list, or video.')
    parser.add_argument('-s', '--savefolder', default='inference_test/out_data', type=str,
                        help='Path to the output directory where results (obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Set device, cpu for using CPU.')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to crop input image; set false only when the test image is well cropped.')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='Sample images from video data for every step.')
    parser.add_argument('--detector', default='fan', type=str,
                        help='Detector for cropping face; check decalib/detectors.py for details.')
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='Rasterizer type: pytorch3d or standard.')
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to render results in original image size; works only when rasterizer_type=standard.')
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use FLAME texture model to generate uv texture map; set True only if you downloaded texture model.')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to extract texture from input image as the uv texture map; set false if you want albedo map from FLAME model.')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save visualization of output.')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save 2D and 3D keypoints.')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save depth image.')
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save outputs as .obj; detail mesh will end with _detail.obj. Note that saving objs could be slow.')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save outputs as .mat.')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save visualization output as separate images.')
    parser.add_argument('--use_mica', default=True, action="store_true",
                        help="Whether to use ArcFace backbone for inference.")
