# FPN

### resnet.py

```
#Bottom-top layers
self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
self.RCNN_layer1 = nn.Sequential(resnet.layer1)
self.RCNN_layer2 = nn.Sequential(resnet.layer2)
self.RCNN_layer3 = nn.Sequential(resnet.layer3)
self.RCNN_layer4 = nn.Sequential(resnet.layer4)

# Top layer
self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

# Lateral layers
self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

# Smooth layers
self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

# ROI Pool feature downsampling
self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

self.RCNN_top = nn.Sequential(
  nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
  nn.ReLU(True),
  nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
  nn.ReLU(True)
  )
```

Top layers，Lateral layers，Smooth layers构成Top-down结构

### fpn.py

```
# feed image data to base model to obtain base feature map
# Bottom-up
c1 = self.RCNN_layer0(im_data)
c2 = self.RCNN_layer1(c1)
c3 = self.RCNN_layer2(c2)
c4 = self.RCNN_layer3(c3)
c5 = self.RCNN_layer4(c4)
# Top-down
p5 = self.RCNN_toplayer(c5)
p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
p4 = self.RCNN_smooth1(p4)
p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
p3 = self.RCNN_smooth2(p3)
p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
p2 = self.RCNN_smooth3(p2)

p6 = self.maxpool2d(p5)

rpn_feature_maps = [p2, p3, p4, p5, p6]
mrcnn_feature_maps = [p2, p3, p4, p5]
```

搭建FPN结构，rpn_feature_maps用于RPN，mrcnn_feature_maps用于RoIPooling。

```
def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
    ''' roi pool on pyramid feature maps'''
    # do roi pooling based on predicted rois
    img_area = im_info[0][0] * im_info[0][1]
    h = rois.data[:, 4] - rois.data[:, 2] + 1
    w = rois.data[:, 3] - rois.data[:, 1] + 1
    roi_level = torch.log(torch.sqrt(h * w) / 224.0)
    roi_level = torch.round(roi_level + 4)
    roi_level[roi_level < 2] = 2
    roi_level[roi_level > 5] = 5
    # roi_level.fill_(5)
    if cfg.POOLING_MODE == 'crop':
        # pdb.set_trace()
        # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        # NOTE: need to add pyrmaid
        grid_xy = _affine_grid_gen(rois, base_feat.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        roi_pool_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

    elif cfg.POOLING_MODE == 'align':
        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / im_info[0][0]
            feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
            roi_pool_feats.append(feat)
        roi_pool_feat = torch.cat(roi_pool_feats, 0)
        box_to_level = torch.cat(box_to_levels, 0)
        idx_sorted, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]

    elif cfg.POOLING_MODE == 'pool':
        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / im_info[0][0]
            feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
            roi_pool_feats.append(feat)
        roi_pool_feat = torch.cat(roi_pool_feats, 0)
        box_to_level = torch.cat(box_to_levels, 0)
        idx_sorted, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]

    return roi_pool_feat
```

输入为mrcnn_feature_maps = [p2, p3, p4, p5]，rois，im_info