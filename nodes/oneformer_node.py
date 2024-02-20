from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
import numpy as np
import time
import cv2

class OneFormerSegmentation:
    def __init__(self, load_immediately=True, device="cpu", high_quality=True):
        # Initialize the segmentation
        # Args:
        #   load_immediately (bool) - Should the model load when this object is constructed
        #   device (str) - "cuda" or "cpu"
        #   high_quality (bool) - Use highest quality segmentation?
        self.device = device
        self.high_quality = high_quality
        if load_immediately:
            self.load()

    def load(self):
        # Load the OneFormer model into memory
        print("Loading OneFormer into memory")

        # Measure performance
        start_time = time.time()

        # Load
        if self.high_quality:
            self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
            self.model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        else:
            self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
            self.model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        # Print performance
        print(f"OneFormer loading time: {time.time() - start_time}s")

    def segment(self, input_img: np.array) -> np.array:
        # Segment an image
        # Args:
        #   input_img (np.array) - RGB image
        # Returns: (np.array) Per-pixel predictions after argmax is applied
        
        # Move model to GPU
        self.model = self.model.to(self.device)

        # Measure performance
        start_time = time.time()

        # Perform segmentation
        inputs = self.processor(input_img, ["semantic"], return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[[input_img.shape[0], input_img.shape[1]]]
        )[0]

        # Print performance
        print(f"OneFormer segmentation time: {time.time() - start_time}s")

        return predicted_semantic_map

    def visualize(self, sem_seg: np.array):
        # Create a segmentation image
        # Args:
        #   sem_seg (np.array): [w,h] 2D image where each coordinate is labeled
        #   colors (dict): Dictionary of colors to use for the colorized output
        # Returns: (torch.Tensor) 2D colorized image of the semantic segmentation
        image = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8)
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.cpu().numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        label2color = np.asarray(list(self.colors().values()))
        for label in labels:
            mask_color = label2color[label]
            binary_mask = (sem_seg == label)
            image[binary_mask] = mask_color
            #cv2.imwrite(f"mask{label}.png", binary_mask*255)
        return image

    def colors(self) -> dict:
        # Retrieve color map
        # Returns (dict): labels and their associated colors
        return {
            'wall': (120, 120, 120),
            'building;edifice': (180, 120, 120),
            'sky': (6, 230, 230),
            'floor;flooring': (80, 50, 50),
            'tree': (4, 200, 3),
            'ceiling': (120, 120, 80),
            'road;route': (140, 140, 140),
            'bed': (204, 5, 255),
            'windowpane;window': (230, 230, 230),
            'grass': (4, 250, 7),
            'cabinet': (224, 5, 255),
            'sidewalk;pavement': (235, 255, 7),
            'person;individual;someone;somebody;mortal;soul': (150, 5, 61),
            'earth;ground': (120, 120, 70),
            'door;double;door': (8, 255, 51),
            'table': (255, 6, 82),
            'mountain;mount': (143, 255, 140),
            'plant;flora;plant;life': (204, 255, 4),
            'curtain;drape;drapery;mantle;pall': (255, 51, 7),
            'chair': (204, 70, 3),
            'car;auto;automobile;machine;motorcar': (0, 102, 200),
            'water': (61, 230, 250),
            'painting;picture': (255, 6, 51),
            'sofa;couch;lounge': (11, 102, 255),
            'shelf': (255, 7, 71),
            'house': (255, 9, 224),
            'sea': (9, 7, 230),
            'mirror': (220, 220, 220),
            'rug;carpet;carpeting': (255, 9, 92),
            'field': (112, 9, 255),
            'armchair': (8, 255, 214),
            'seat': (7, 255, 224),
            'fence;fencing': (255, 184, 6),
            'desk': (10, 255, 71),
            'rock;stone': (255, 41, 10),
            'wardrobe;closet;press': (7, 255, 255),
            'lamp': (224, 255, 8),
            'bathtub;bathing;tub;bath;tub': (102, 8, 255),
            'railing;rail': (255, 61, 6),
            'cushion': (255, 194, 7),
            'base;pedestal;stand': (255, 122, 8),
            'box': (0, 255, 20),
            'column;pillar': (255, 8, 41),
            'signboard;sign': (255, 5, 153),
            'chest;of;drawers;chest;bureau;dresser': (224, 5, 255),
            'counter': (235, 12, 255),
            'sand': (160, 150, 20),
            'sink': (0, 163, 255),
            'skyscraper': (140, 140, 140),
            'fireplace;hearth;open;fireplace': (250, 10, 15),
            'refrigerator;icebox': (20, 255, 0),
            'grandstand;covered;stand': (31, 255, 0),
            'path': (255, 31, 0),
            'stairs;steps': (255, 224, 0),
            'runway': (153, 255, 0),
            'case;display;case;showcase;vitrine': (0, 0, 255),
            'pool;table;billiard;table;snooker;table': (255, 71, 0),
            'pillow': (0, 235, 255),
            'screen;door;screen': (0, 173, 255),
            'stairway;staircase': (31, 0, 255),
            'river': (11, 200, 200),
            'bridge;span': (255 ,82, 0),
            'bookcase': (0, 255, 245),
            'blind;screen': (0, 61, 255),
            'coffee;table;cocktail;table': (0, 255, 112),
            'toilet;can;commode;crapper;pot;potty;stool;throne': (0, 255, 133),
            'flower': (255, 0, 0),
            'book': (255, 163, 0),
            'hill': (255, 102, 0),
            'bench': (194, 255, 0),
            'countertop': (0, 143, 255),
            'stove;kitchen;stove;range;kitchen;range;cooking;stove': (51, 255, 0),
            'palm;palm;tree': (0, 82, 255),
            'kitchen;island': (224, 5, 255),
            'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system': (0, 255, 173),
            'swivel;chair': (10, 0, 255),
            'boat': (173, 255, 0),
            'bar': (0, 255, 153),
            'arcade;machine': (255, 92, 0),
            'hovel;hut;hutch;shack;shanty': (255, 0, 255),
            'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle': (255, 0, 245),
            'towel': (255, 0, 102),
            'light;light;source': (255, 173, 0),
            'truck;motortruck': (255, 0, 20),
            'tower': (255, 184, 184),
            'chandelier;pendant;pendent': (0, 31, 255),
            'awning;sunshade;sunblind': (0, 255, 61),
            'streetlight;street;lamp': (0, 71, 255),
            'booth;cubicle;stall;kiosk': (255, 0, 204),
            'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box': (0, 255, 194),
            'airplane;aeroplane;plane': (0, 255, 82),
            'dirt;track': (0, 10, 255),
            'apparel;wearing;apparel;dress;clothes': (0, 112, 255),
            'pole': (51, 0, 255),
            'land;ground;soil': (0, 194, 255),
            'bannister;banister;balustrade;balusters;handrail': (0, 122, 255),
            'escalator;moving;staircase;moving;stairway': (0, 255, 163),
            'ottoman;pouf;pouffe;puff;hassock': (255, 153, 0),
            'bottle': (0, 255, 10),
            'buffet;counter;sideboard': (255, 112, 0),
            'poster;posting;placard;notice;bill;card': (143, 255, 0),
            'stage': (82, 0, 255),
            'van': (163, 255, 0),
            'ship': (255, 235, 0),
            'fountain': (8, 184, 170),
            'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter': (133, 0, 255),
            'canopy': (0, 255, 92),
            'washer;automatic;washer;washing;machine': (184, 0, 255),
            'plaything;toy': (255, 0, 31),
            'swimming;pool;swimming;bath;natatorium': (0, 184, 255),
            'stool': (0, 214, 255),
            'barrel;cask': (255, 0, 112),
            'basket;handbasket': (92, 255, 0),
            'waterfall;falls': (0, 224, 255),
            'tent;collapsible;shelter': (112, 224, 255),
            'bag': (70, 184, 160),
            'minibike;motorbike': (163, 0, 255),
            'cradle': (153, 0, 255),
            'oven': (71, 255, 0),
            'ball': (255, 0, 163),
            'food;solid;food': (255, 204, 0),
            'step;stair': (255, 0, 143),
            'tank;storage;tank': (0, 255, 235),
            'trade;name;brand;name;brand;marque': (133, 255, 0),
            'microwave;microwave;oven': (255, 0, 235),
            'pot;flowerpot': (245, 0, 255),
            'animal;animate;being;beast;brute;creature;fauna': (255, 0, 122),
            'bicycle;bike;wheel;cycle': (255, 245, 0),
            'lake': (10, 190, 212),
            'dishwasher;dish;washer;dishwashing;machine': (214, 255, 0),
            'screen;silver;screen;projection;screen': (0, 204, 255),
            'blanket;cover': (20, 0, 255),
            'sculpture': (255, 255, 0),
            'hood;exhaust;hood': (0, 153, 255),
            'sconce': (0, 41, 255),
            'vase': (0, 255, 204),
            'traffic;light;traffic;signal;stoplight': (41, 0, 255),
            'tray': (41, 255, 0),
            'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin': (173, 0, 255),
            'fan': (0, 245, 255),
            'pier;wharf;wharfage;dock': (71, 0, 255),
            'crt;screen': (122, 0, 255),
            'plate': (0, 255, 184),
            'monitor;monitoring;device': (0, 92, 255),
            'bulletin;board;notice;board': (184, 255, 0),
            'shower': (0, 133, 255),
            'radiator': (255, 214, 0),
            'glass;drinking;glass': (25, 194, 194),
            'clock': (102, 255, 0),
            'flag': (92, 0, 255),
        }

class OneFormerNode:
    """
    OneFormer ADE20k segmentation
    """

    # Singletons
    oneformer = None 

    def __init__(self):
        print("OneFormer init called")
        if OneFormerNode.oneformer == None:
            OneFormerNode.oneformer = OneFormerSegmentation()
            self.oneformer = OneFormerNode.oneformer
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
                "high_quality": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "EliTest"

    def process(self, image, high_quality):
        print("Executing OneFormer")
        if bool(high_quality) != self.oneformer.high_quality:
            self.oneformer.high_quality = bool(high_quality)
            self.oneformer.load()    
        input_img = (image * 255.0).squeeze(0).to(torch.uint8)
        preds = self.oneformer.segment(input_img)
        image = self.oneformer.visualize(preds)
        output_img = (torch.from_numpy(image) / 255.0).unsqueeze(0)
        return (output_img,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "eliTestSeg": OneFormerNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "eliTestSeg": "OneFormer Node"
}
