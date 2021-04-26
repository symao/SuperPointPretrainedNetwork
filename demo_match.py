import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
from demo_superpoint import *

if __name__ == '__main__':
  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=2,
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--write', action='store_true',
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')
  parser.add_argument('--anchor', type=str, default=None, help='Anchor image path')
  opt = parser.parse_args()
  print(opt)

  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)
  img, ret = vs.next_frame()
  assert ret, 'Error when reading the first frame (try different --input?)'

  if opt.anchor is not None and os.path.exists(opt.anchor):
    img = cv2.resize(cv2.imread(opt.anchor, cv2.IMREAD_GRAYSCALE),(opt.W, opt.H)).astype('float32')/255.0

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

  # anchor image descriptor
  anchor_pts, anchor_desc, anchor_heatmap = fe.run(img)

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4
  anchor_bgr = cv2.cvtColor((img*255.).astype('uint8'), cv2.COLOR_GRAY2BGR)
  for pt in anchor_pts.T:
    pt1 = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(anchor_bgr, pt1, 1, (0, 255, 0), -1, lineType=16)
  cv2.putText(anchor_bgr, 'Anchor', font_pt, font, font_sc, font_clr, lineType=16)

  writer = None

  while True:
    # Get a new image.
    img, status = vs.next_frame()
    bgr = cv2.cvtColor((img*255.).astype('uint8'), cv2.COLOR_GRAY2BGR)
    if status is False:
      break
    # Get points and descriptors.
    start1 = time.time()
    pts, desc, heatmap = fe.run(img)
    end1 = time.time()
    matches = None
    if desc is not None:
        matches = tracker.nn_match_two_way(anchor_desc, desc, opt.nn_thresh)

    for pt in pts.T:
      pt1 = (int(round(pt[0])), int(round(pt[1])))
      cv2.circle(bgr, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(bgr, 'Current frame', font_pt, font, font_sc, font_clr, lineType=16)
    img_draw = np.hstack((anchor_bgr, bgr))
    if matches is not None and matches.shape[1] > 0:
      for id1,id2,score in matches.T:
        pt1 = anchor_pts[:2, int(id1)]
        pt2 = pts[:2, int(id2)]
        pt2[0] += opt.W
        pt1 = tuple([int(x) for x in pt1])
        pt2 = tuple([int(x) for x in pt2])
        cv2.line(img_draw, pt1, pt2, (0,0,255), 1)

    cv2.imshow('img', img_draw)
    # if writer is None:
    #     writer = cv2.VideoWriter("./demo.mp4",cv2.VideoWriter_fourcc(*'XVID'), 20, img_draw.shape[:2][::-1])
    # writer.write(img_draw)

    if heatmap is not None:
      min_conf = 0.001
      heatmap[heatmap < min_conf] = min_conf
      heatmap = -np.log(heatmap)
      heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
      out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
      out3 = (out3*255).astype('uint8')
      cv2.imshow('out3', out3)

    key = cv2.waitKey(10)
    if key == 27:
        break
  if writer:
    writer.release()
  print('==> Finshed Demo.')
