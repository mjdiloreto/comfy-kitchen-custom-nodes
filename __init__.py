from .nodes import \
    oneformer_node, \
    resize_image_node
NODE_CLASS_MAPPINGS = {
    **oneformer_node.NODE_CLASS_MAPPINGS,
    **resize_image_node.NODE_CLASS_MAPPINGS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **oneformer_node.NODE_DISPLAY_NAME_MAPPINGS,
    **resize_image_node.NODE_DISPLAY_NAME_MAPPINGS
}
