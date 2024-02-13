import torch
import numpy as np
import time
import cv2

class ResizeImageNode:
    """
    Resize an image to fit a max resolution
    """

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "max_res": ("INT", {
                    "default": "768"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Output",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "EliTest"

    def process(self, image, max_res):
        print("Executing Resize Image Node")
        
        input_img = (image * 255.0).clip(0,255).squeeze(0)
        input_img = input_img.cpu().numpy()

        # Determine the GML resolution
        w = input_img.shape[1]
        h = input_img.shape[0]
        if w > h:
            aspect = h / w
            gml_width = max_res
            gml_height = int(gml_width * aspect)
        else:
            aspect = w / h
            gml_height = max_res
            gml_width = int(gml_height * aspect)

        image = cv2.resize(input_img, dsize=(gml_width, gml_height), interpolation=cv2.INTER_CUBIC)
        
        output_img = (torch.from_numpy(image) / 255.0).unsqueeze(0)
        return (output_img,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "EliResizeImage": ResizeImageNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "EliResizeImage": "ResizeImage Node"
}