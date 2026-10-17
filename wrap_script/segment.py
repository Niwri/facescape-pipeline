import nersemble_paths
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import cv2

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab

from model import U2NET


class BackgroundSegmentation:

    def __init__(self, checkpoint_path=None):
        model_name='u2net'
        model_dir = checkpoint_path or os.path.join("/scratch", "ondemand32", "irwinngo", 'models', model_name + '_human_seg.pth')

        if(model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            self.net = U2NET(3,1)

        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.net.eval()

        
        

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    def get_mask(self, image):
        """Returns the U2Net binary foreground mask (0/255, HxW) for image."""
        transformed = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])({
            'imidx': np.array([0]),
            'image': image,
            'label': np.zeros_like(image)  # or use None if your transform supports it
        })
        inputs_test = transformed['image'].unsqueeze(0)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= self.net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = self.normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        mask = (predict_np * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)

        del d1,d2,d3,d4,d5,d6,d7
        return binary_mask

    def remove_background(self, image):
        binary_mask = self.get_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
        return masked_image

class FaceSegmentation:

    def __init__(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or str(nersemble_paths.MODELS_ROOT / "sam_vit_h_4b8939.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "default"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)

        self.predictor = SamPredictor(self.sam)

    def get_face_mask(self, image, point=None, points=None, show_plot=False):
        """Returns SAM's boolean face mask (HxW), prompted from one or more
        points (all treated as positive/foreground).

        point: (x, y) pixel to prompt SAM with (single-point case). Defaults
            to the image center if neither `point` nor `points` given.
        points: list of (x, y) pixels for a multi-point prompt -- takes
            priority over `point` if both are given.
        """
        h, w = image.shape[:2]
        if points is not None:
            input_point = np.array(points)
            input_label = np.ones(len(points), dtype=int)
        else:
            if point is None:
                point = (w // 2, h // 2)
            input_point = np.array([point])
            input_label = np.array([1])

        self.predictor.set_image(image)

        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        mask = masks[0]

        if(show_plot):
            # Import plotting only when requested to avoid Qt init in headless runs.
            import matplotlib.pyplot as plt
            masked_image = image.copy()
            masked_image[~mask] = 0
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(image)
            ax2.imshow(masked_image)
            plt.show()
        return mask
