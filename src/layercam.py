"""
Created on Mon Jul 5 12:39:11 2021

@author: Peng-Tao Jiang - github.com/PengtaoJiang
"""
import numpy as np
import torch
from PIL import Image


class LayerCam():
    """
        Produces class activation map
    """

    def __init__(self, model, extractor):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = extractor
        self.name = "LayerCam"

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(model_output.device)
        one_hot_output[target_class][0] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradient
        guided_gradients = self.extractor.gradients.cpu().data.numpy()
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()
        # Get weights from gradients
        weights = guided_gradients
        weights[weights < 0] = 0  # discard negative gradients
        # Element-wise multiply the weight with its conv output and then, sum
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

        # can't resize third dimension with 3 color channels, so bring to gray
        if cam.shape[-1] % 3 != 0:
            cam = cam.reshape((cam.shape[0], -1))

        cam = cam.astype(np.uint8)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255

        return cam
