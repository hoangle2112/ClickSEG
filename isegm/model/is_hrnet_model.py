import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps
from isegm.utils.crop_local import  map_point_in_bbox,get_focus_cropv1, get_focus_cropv2, get_object_crop, get_click_crop
from isegm.inference.transforms import SigmoidForPred, ResizeTrans, LimitLongestSide

class HRNetModel(ISModel):
    @serialize
    def __init__(self, width=48, ocr_width=256, small=True, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version = 's1',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))
        
        self.feature_extractor.apply(LRMult(2))
        self.width=width
        self.pipeline_version = pipeline_version
        if self.pipeline_version == 's1':
            base_radius = 2
        else:
            base_radius = 5
        
        self.refiner = RefineLayer(feature_dims=ocr_width * 2)
        
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        
        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        
        self.crop_l = 256
        self.transforms = []
        self.transforms.append(LimitLongestSide(800))
        self.transforms.append(ResizeTrans(256))
        self.transforms.append(SigmoidForPred())
        self.net_clicks_limit = 128
        self.device = "cuda:0"
                                    
    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features
    
    def backbone_forward(self, image, coord_features=None):
        mask, mask_aux, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask_aux, 'feature': feature}

    def refine(self, cropped_image, cropped_points, full_feature, full_logits, bboxes):
        '''
        bboxes : [b,5]
        '''
        h1 = cropped_image.shape[-1]
        h2 = full_feature.shape[-1]
        r = h1/h2

        cropped_feature = roi_align(full_feature,bboxes,full_feature.size()[2:], spatial_scale=1/r, aligned = True)
        cropped_logits = roi_align(full_logits,bboxes,cropped_image.size()[2:], spatial_scale=1, aligned = True)
        
        click_map = self.dist_maps_refine( cropped_image,cropped_points)
        refined_mask, trimap = self.refiner(cropped_image,click_map,cropped_feature,cropped_logits)
        return {'instances_refined': refined_mask, 'instances_coarse':cropped_logits, 'trimap':trimap}
    
    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed
    
    def mapp_roi(self, focus_roi, global_roi):
        yg1,yg2,xg1,xg2 = global_roi
        hg,wg = yg2-yg1, xg2-xg1
        yf1,yf2,xf1,xf2 = focus_roi
        
        yf1_n = (yf1-yg1) * (self.crop_l/hg)
        yf2_n = (yf2-yg1) * (self.crop_l/hg)
        xf1_n = (xf1-xg1) * (self.crop_l/wg)
        xf2_n = (xf2-xg1) * (self.crop_l/wg)

        yf1_n = max(yf1_n,0)
        yf2_n = min(yf2_n,self.crop_l)
        xf1_n = max(xf1_n,0)
        xf2_n = min(xf2_n,self.crop_l)
        return (yf1_n,yf2_n,xf1_n,xf2_n)
    
    def _get_refine(self, coarse_mask, image, clicks, feature, focus_roi, focus_roi_in_global_roi):
        y1,y2,x1,x2 = focus_roi
        image_focus = image[:,:,y1:y2,x1:x2]
        image_focus = F.interpolate(image_focus,(self.crop_l,self.crop_l),mode='bilinear',align_corners=True)
        mask_focus = coarse_mask
        points_nd = self.get_points_nd_inbbox(clicks,y1,y2,x1,x2)
        y1,y2,x1,x2 = focus_roi_in_global_roi
        roi = torch.tensor([0,x1, y1, x2, y2]).unsqueeze(0).float().to(image_focus.device)

        pred = self.refine(image_focus,points_nd, feature, mask_focus, roi) #['instances_refined'] 
        focus_coarse, focus_refined = pred['instances_coarse'] , pred['instances_refined'] 
        self.focus_coarse = torch.sigmoid(focus_coarse).cpu().numpy()[0, 0] * 255
        self.focus_refined = torch.sigmoid(focus_refined).cpu().numpy()[0, 0] * 255
        return focus_refined
    
    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)
    
    def get_points_nd_inbbox(self, clicks_list, y1,y2,x1,x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg =len(clicks_list) - num_pos 
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [],[]
        for click in clicks_list:
            flag,y,x,index = click.is_positive, click.coords[0],click.coords[1], 0
            y,x = map_point_in_bbox(y,x,y1,y2,x1,x2,self.crop_l)
            if flag:
                pos_clicks.append( (y,x,index))
            else:
                neg_clicks.append( (y,x,index) )

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def forward(self, image, points):
        points_raw = points.copy()
        image_full, prev_mask_full = self.prepare_input(image)

        image, points_transformed, is_image_changed = self.apply_transforms(
            image, [points]
        )
        points = self.get_points_nd(points_transformed)
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:,1:,:,:]

        if self.pipeline_version == 's1':
            small_image = F.interpolate(image, scale_factor=0.5,mode='bilinear',align_corners=True)
            small_coord_features = F.interpolate(coord_features, scale_factor=0.5,mode='bilinear',align_corners=True)
        else:
            small_image = image
            small_coord_features = coord_features

        small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward( small_image, small_coord_features)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)

        # Move to HRNetModel
        prediction = F.interpolate(outputs['instances'], mode='bilinear', align_corners=True, size=image.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        prediction  = torch.log( prediction/(1-prediction)  )
        coarse_mask = prediction
        
        coarse_mask_np = coarse_mask.cpu().numpy()[0, 0] 
        prev_mask_np = prev_mask_full.cpu().numpy()[0, 0] 

        h,w = prev_mask_full.shape[-2],prev_mask_full.shape[-1]
        global_roi = (0,h,0,w)  
        click = points_raw[-1]
        last_y,last_x = click.coords[0],click.coords[1]

        # # === Ablation Studies for Different Strategies of Focus Crop
        # y1,y2,x1,x2 = get_focus_cropv1(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, self.focus_crop_r)
        y1,y2,x1,x2 = get_focus_cropv1(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.2)
        # #y1,y2,x1,x2 = get_focus_cropv1(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.6)
        # #y1,y2,x1,x2 = get_focus_cropv2(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)
        # #y1,y2,x1,x2 = get_object_crop(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)
        # #y1,y2,x1,x2 = get_click_crop(coarse_mask_np,prev_mask_np, global_roi,last_y,last_x, 1.4)
        
        focus_roi = (y1,y2,x1,x2)
        # self.focus_roi = focus_roi
        focus_roi_in_global_roi = self.mapp_roi(focus_roi, global_roi)
        focus_pred = self._get_refine(outputs['instances'],image_full,points_raw, outputs['feature'], focus_roi, focus_roi_in_global_roi)#.cpu().numpy()[0, 0]
        focus_pred = F.interpolate(focus_pred,(y2-y1,x2-x1),mode='bilinear',align_corners=True)#.cpu().numpy()[0, 0]
        
        # if len(points_raw) > 10:
        #     coarse_mask = prev_mask
        #     coarse_mask  = torch.log( coarse_mask/(1-coarse_mask)  )
        coarse_mask[:,:,y1:y2,x1:x2] =  focus_pred
        return coarse_mask
        
class RefineLayer(nn.Module):
    """
    Refine Layer for Full Resolution
    """
    def __init__(self, input_dims = 6, mid_dims = 32, feature_dims = 96, num_classes = 1,  **kwargs):
        super(RefineLayer, self).__init__()
        self.num_classes = num_classes
        self.image_conv1 = ConvModule(
            in_channels=input_dims,
            out_channels= mid_dims,
            kernel_size=3,
            stride=2,
            padding=1,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        self.image_conv2 = XConvBnRelu( mid_dims, mid_dims)
        self.image_conv3 = XConvBnRelu( mid_dims, mid_dims)
        

        self.refine_fusion = ConvModule(
            in_channels= feature_dims,
            out_channels= mid_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        
        self.refine_fusion2 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_fusion3 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_pred = nn.Conv2d( mid_dims, num_classes,3,1,1)
        self.refine_trimap = nn.Conv2d( mid_dims, num_classes,3,1,1)
    

    def forward(self, input_image, click_map, final_feature, cropped_logits):
        #cropped_logits = cropped_logits.clone().detach()
        #final_feature = final_feature.clone().detach()

        mask = cropped_logits #resize(cropped_logits, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        bin_mask = torch.sigmoid(mask) #> 0.49
        input_image = torch.cat( [input_image,click_map,bin_mask], 1)

        final_feature = self.refine_fusion(final_feature)
        image_feature = self.image_conv1(input_image)
        image_feature = self.image_conv2(image_feature)
        image_feature = self.image_conv3(image_feature)
        pred_feature = resize(final_feature, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        #fusion_gate = self.feature_gate(final_feature)
        #fusion_gate = resize(fusion_gate, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        pred_feature = pred_feature + image_feature #* fusion_gate

        pred_feature = self.refine_fusion2(pred_feature)
        pred_feature = self.refine_fusion3(pred_feature)
        pred_full = self.refine_pred(pred_feature)
        trimap = self.refine_trimap(pred_feature)
        trimap = F.interpolate(trimap, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        pred_full = F.interpolate(pred_full, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        trimap_sig = torch.sigmoid(trimap)
        pred = pred_full * trimap_sig + mask * (1-trimap_sig)
        return pred, trimap

class ConvModule(nn.Module):
    def __init__(self, in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
       

    def forward(self, x):
        return self.activation( self.norm( self.conv(x)  ) )




class XConvBnRelu(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16,   **kwargs):
        super(XConvBnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.activation(x)
        return x





def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)