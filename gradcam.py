"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for explainable CNN dermatological image classification.

Provides robust, differentiable activation map extraction, safe normalization,
and clinical research explainability overlays.
"""

from typing import Tuple, Optional, Union
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def find_target_conv_layer(model: tf.keras.Model) -> Tuple[Optional[tf.keras.layers.Layer], Optional[tf.keras.Model]]:
    """
    Locates the most appropriate convolutional / activation layer for Grad-CAM.
    Supports both nested architectures (e.g. MobileNetV2 inside Sequential/Functional)
    and flattened functional models.
    """
    # 1. Search for nested backbone
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or (hasattr(layer, "layers") and len(layer.layers) > 10):
            backbone = layer
            break

    search_target = backbone if backbone is not None else model

    # Look for known MobileNetV2 final activation layers
    for candidate_name in ["out_relu", "Conv_1", "Conv_1_bn"]:
        try:
            target_layer = search_target.get_layer(candidate_name)
            return target_layer, backbone
        except ValueError:
            continue

    # Fallback: Find the last Conv2D or Relu layer
    for layer in reversed(search_target.layers):
        layer_type = layer.__class__.__name__.lower()
        if "conv" in layer_type or "relu" in layer_type:
            return layer, backbone

    return None, backbone


def compute_gradcam(
    model: tf.keras.Model,
    preprocessed_img: np.ndarray,
    class_idx: Optional[int] = None,
    target_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes the 2D Grad-CAM heatmap for a given input tensor and class index.
    
    Args:
        model: Trained Keras model.
        preprocessed_img: Input tensor with shape (1, 224, 224, 3).
        class_idx: Target class index. If None, uses top predicted class.
        target_layer_name: Specific layer name, or None to auto-detect.

    Returns:
        2D numpy array with values in [0.0, 1.0], shape (H, W) corresponding to conv layer.
    """
    if len(preprocessed_img.shape) == 3:
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    target_layer, backbone = find_target_conv_layer(model)
    if target_layer_name is not None:
        try:
            target_layer = (backbone or model).get_layer(target_layer_name)
        except ValueError:
            pass

    if target_layer is None:
        raise RuntimeError("Could not identify a valid convolutional feature layer for Grad-CAM.")

    # Determine graph architecture and compute gradients
    if backbone is not None:
        # Deconstruct nested graph into feature extractor and classification head
        feature_extractor = tf.keras.Model(
            inputs=backbone.input,
            outputs=[target_layer.output, backbone.output]
        )
        # Reconstruct classification head from remaining layers
        head_input = tf.keras.Input(shape=backbone.output.shape[1:])
        x = head_input
        found_backbone = False
        for l in model.layers:
            if l == backbone:
                found_backbone = True
                continue
            if found_backbone:
                x = l(x)
        classifier_head = tf.keras.Model(head_input, x)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = feature_extractor(preprocessed_img, training=False)
            tape.watch(conv_outputs)
            predictions = classifier_head(backbone_features, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
    else:
        # Direct functional graph
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_img, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError(
            f"Gradients could not be computed for layer '{target_layer.name}'. "
            "Ensure the layer is differentiable and connected to the output."
        )

    # Compute channel-wise mean gradient (global average pooling of gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolutional feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to focus strictly on features positively influencing the target class
    cam = tf.maximum(cam, 0.0)

    # Normalize safely
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val
    else:
        # Zero-valued activation fallback
        cam = tf.zeros_like(cam)

    return cam.numpy()


def overlay_gradcam(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes heatmap, colorizes it, and blends it with the original image.
    
    Args:
        original_image: PIL Image or RGB uint8 numpy array.
        heatmap: 2D numpy array [0, 1].
        alpha: Heatmap blend opacity (0.0 to 1.0).
        colormap_name: Matplotlib colormap (e.g. 'jet', 'turbo', 'inferno').

    Returns:
        Tuple of (colored_heatmap_rgb, blended_overlay_rgb).
    """
    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image.convert("RGB"))
    else:
        orig_np = original_image.copy()

    h, w = orig_np.shape[:2]

    # Resize 2D heatmap to original image dimensions using PIL bilinear interpolation
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Colorize using Matplotlib colormap (compatible with Matplotlib >= 3.8)
    import matplotlib as mpl
    try:
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        cmap = mpl.cm.get_cmap(colormap_name) if hasattr(mpl.cm, "get_cmap") else mpl.colormaps["jet"]
    colored_map = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel -> RGB in [0, 1]
    colored_map = (colored_map * 255).astype(np.uint8)

    # Superimpose heatmap onto original image
    blended = np.clip(alpha * colored_map + (1.0 - alpha) * orig_np, 0, 255).astype(np.uint8)

    return colored_map, blended
