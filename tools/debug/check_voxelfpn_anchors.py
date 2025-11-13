#!/usr/bin/env python3
import yaml
from pathlib import Path

cfg_path = Path(__file__).resolve().parents[1] / 'cfgs' / 'kitti_models' / 'voxel_fpn.yaml'
data = yaml.safe_load(cfg_path.read_text())

print('Config file:', cfg_path)
ag = data['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']
nms = data['MODEL']['POST_PROCESSING'].get('NMS_THRESH', None)
print(f'NMS_THRESH: {nms}')
for cfg in ag:
    name = cfg.get('class_name', 'unknown')
    sizes = cfg.get('anchor_sizes', [])
    bottoms = cfg.get('anchor_bottom_heights', [])
    print(f'Class: {name}')
    for s in sizes:
        h = s[2]
        for b in bottoms:
            center = b + h / 2.0
            print(f'  size(h)={h:.3f} bottom={b:.3f} -> center={center:.3f}')

print('\nQuick checks done.')
