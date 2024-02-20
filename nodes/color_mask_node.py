import torch
import numpy as np
import time
import cv2

class ColorMaskNode:
    """
    Convert one or more colors in an image into a mask
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
                "hex_colors": ("STRING", {
                    "multiline": True, 
                    "default": "#000000, #ffffff"
                }),
                "tolerance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255
                })
            },
        }

    RETURN_TYPES = ("MASK",)
    #RETURN_NAMES = ("output",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "EliTest"

    def process(self, image, hex_colors, tolerance):
        print("Executing Color Mask")
        
        color_names = hex_colors.replace(" ", "").split(",")
        colors = list(map(lambda x: self.hex_to_rgb(x), color_names))
        
        input_img = (image * 255.0).squeeze(0)
        input_img = input_img.cpu().numpy()
        image = self.create_mask(input_img, colors, tolerance)
        
        output_img = (torch.from_numpy(image) / 255.0).unsqueeze(0)

        channel = 0
        mask = output_img[0, :, :, channel]

        return (mask,)

    def hex_to_rgb(self, value):
        # Determine the RGB for a given HEX value
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def create_mask(self, input_img, colors, tolerance):
        # Create a mask image
        image = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
        for i, c in enumerate(colors):
            t = tolerance
            min_color = (c[0] - t, c[1] - t, c[2] - t)
            max_color = (c[0] + t, c[1] + t, c[2] + t)
            bmask = cv2.inRange(input_img, min_color, max_color)
            image += bmask
            #cv2.imwrite(f"mask{i}.png", bmask*255)
        image = image.clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "eliColorMask": ColorMaskNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "eliColorMask": "ColorMask Node"
}