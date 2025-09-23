"""
obs_transforms.py

Contains observation-level transforms used in the orca data pipeline.

These transforms operate on the "observation" dictionary, and are applied at a per-frame level.
"""

from typing import Dict, Tuple, Union, Optional, List
from functools import partial

import dlimp as dl
import tensorflow as tf
import cv2

from absl import logging
import math
import numpy as np
from dlimp.augmentations import AUGMENT_OPS

def random_rotation(image: tf.Tensor, 
                    angle_range: float,
                    seed: tf.Tensor, 
                    interpolation: str = 'bilinear',
                    intrinsics: Optional[tf.Tensor] = None,) -> tf.Tensor:
    """random image rotation
    
    Args:
        image: tf.Tensor with shape [H, W, C] or [B, H, W, C]
        seed: tf.Tensor with shape [2]
        angle_range: angle
        interpolation: interpolation method, 'bilinear' or 'nearest'

    Returns:
        rotated image(s)
    """
    angle_min, angle_max = -angle_range, angle_range
    if interpolation == 'bilinear':
        interp_flag = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        interp_flag = cv2.INTER_NEAREST
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")
    
    angle = tf.random.stateless_uniform(
        shape=[], seed=seed, minval=angle_min, maxval=angle_max, dtype=tf.float32
    )
    
    # wrap numpy operation in tf.py_function
    def _rotate_image(image, angle):
        image_np = image.numpy()
        angle_np = angle.numpy()
        h, w = image_np.shape[-3:-1]
        center = (w/2, h/2)
        channels = image_np.shape[-1]
        
        if len(image_np.shape) == 3:
            if channels == 1:
                image_np = image_np.squeeze(-1) # cv2 accepts (h, w) not (h, w, 1) for single channel images
            M = cv2.getRotationMatrix2D(center, angle_np, scale=1.0)
            rotated_np = cv2.warpAffine(
                image_np, M, (w, h),
                flags=interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[0] * channels
            )
            if channels == 1:
                rotated_np = rotated_np[..., None]
            
            return rotated_np
        else:
            batch_size = image_np.shape[0]
            rotated_batch = []
            if channels == 1:
                image_np = image_np.squeeze(-1)  # cv2 accepts (h, w) not (h, w, 1) for single channel images
            for i in range(batch_size):
                M = cv2.getRotationMatrix2D(center, angle_np, scale=1.0)
                rotated_np = cv2.warpAffine(
                    image_np[i], M, (w, h),
                    flags=interp_flag,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=[0] * channels
                )
                if channels == 1:
                    rotated_np = rotated_np[..., None]
                rotated_batch.append(rotated_np)
            return np.stack(rotated_batch)
    
    # wrap numpy operation in tf.py_function
    rotated = tf.py_function(
        _rotate_image,
        [image, angle],
        Tout=image.dtype
    )
    
    # set output shape to make tf graph happy
    if image.shape.ndims == 3:
        rotated.set_shape(image.shape)
    else:
        rotated.set_shape([None] + image.shape.as_list()[1:])
    
    if intrinsics is not None:
        intrinsics = tf.cast(intrinsics, tf.float32)
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
        angle_rad = tf.constant(angle * math.pi / 180.0)
        cos_a, sin_a = tf.cos(angle_rad), tf.sin(angle_rad)
        tx = (1 - cos_a) * cx + sin_a * cy
        ty = (1 - cos_a) * cy - sin_a * cx
        H = tf.stack([
            tf.stack([cos_a, -sin_a, tx], axis=-1),
            tf.stack([sin_a, cos_a, ty], axis=-1),
            tf.constant([0, 0, 1], shape=[1, 3]),
        ], axis=0)
        intrinsics = tf.matmul(H, intrinsics)
        return {"image": rotated, "intrinsics": intrinsics}
    
    return {"image": rotated, "intrinsics": tf.zeros([3, 3])}

def random_resized_crop_depth(image, scale, ratio, seed, intrinsics=None):
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
    batch_size = tf.shape(image)[0]
    # taken from https://keras.io/examples/vision/nnclr/#random-resized-crops
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]

    random_scales = tf.random.stateless_uniform((batch_size,), seed, scale[0], scale[1])
    random_ratios = tf.exp(
        tf.random.stateless_uniform((batch_size,), seed, log_ratio[0], log_ratio[1])
    )

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.stateless_uniform(
        (batch_size,), seed, 0, 1 - new_heights
    )
    width_offsets = tf.random.stateless_uniform((batch_size,), seed, 0, 1 - new_widths)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (height, width), method="nearest"
    )

    if image.shape[0] == 1:
        image = image[0]
    
    if intrinsics is not None:
        # intrinsics: [batch, 3, 3]
        if intrinsics.shape.ndims == 2:
            intrinsics = tf.expand_dims(intrinsics, axis=0)
        K = intrinsics  # shorthand

        # scale
        s_x = 1.0 / new_widths   # [batch]
        s_y = 1.0 / new_heights  # [batch]

        # original fx, fy, cx, cy
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        # new fx, fy, cx, cy
        new_fx = fx * s_x
        new_fy = fy * s_y
        new_cx = (cx - width_offsets * tf.cast(width, tf.float32)) * s_x
        new_cy = (cy - height_offsets * tf.cast(height, tf.float32)) * s_y

        # rebuild intrinsics
        zeros = tf.zeros_like(s_x)
        ones  = tf.ones_like(s_x)
        K_new = tf.stack([
            tf.stack([new_fx, zeros, new_cx], axis=1),
            tf.stack([zeros, new_fy, new_cy], axis=1),
            tf.stack([zeros, zeros, ones ],  axis=1),
        ], axis=1)  # [batch,3,3]

        if K.shape[0] == 1:
            K_new = K_new[0]
        
        return {"image": image, "intrinsics": K_new}
    
    return {"image": image, "intrinsics": tf.zeros([3, 3])}

def random_drop_all_image(image: tf.Tensor, drop_prob: float, seed: tf.Tensor):
    """randomly drop the image with probability drop_prob"""
    drop = tf.random.stateless_uniform((), seed, minval=0, maxval=1, dtype=tf.float32)
    return tf.cond(
        tf.math.less(drop, drop_prob),
        lambda: tf.zeros_like(image),
        lambda: image
    )

def random_drop_all_depth(image: tf.Tensor, drop_prob: float, seed: tf.Tensor):
    """randomly drop the image with probability drop_prob"""
    drop = tf.random.stateless_uniform((), seed, minval=0, maxval=1, dtype=tf.float32)
    return tf.cond(
        tf.math.less(drop, drop_prob),
        lambda: tf.random.stateless_uniform(tf.shape(image), seed, minval=0, maxval=1, dtype=tf.float32),
        lambda: image
    )

def random_mask(image: tf.Tensor, mask_scale: float, seed: tf.Tensor) -> tf.Tensor:
    """randomly mask out a region of the image, with a size determined by mask_scale"""
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    # get mask height and width
    mask_height = tf.random.stateless_uniform((), seed, minval=1, maxval=math.ceil(height*mask_scale), dtype=tf.int32)
    mask_width = tf.random.stateless_uniform((), seed+1, minval=1, maxval=math.ceil(width*mask_scale), dtype=tf.int32)

    # get top-left corner of mask
    mask_top = tf.random.stateless_uniform((), seed+2, minval=0, maxval=tf.shape(image)[0] - mask_height, dtype=tf.int32)
    mask_left = tf.random.stateless_uniform((), seed+3, minval=0, maxval=tf.shape(image)[1] - mask_width, dtype=tf.int32)

    # create mask
    mask = tf.ones((height, width), dtype=image.dtype)
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.stack(tf.meshgrid(tf.range(mask_top, mask_top + mask_height), tf.range(mask_left, mask_left + mask_width)), axis=-1),
        tf.zeros((mask_height, mask_width), dtype=image.dtype)
    )

    mask = tf.expand_dims(mask, axis=-1)  # Expand to match image channels
    return image * mask


AUGMENT_OPS["random_rotation"] = random_rotation
AUGMENT_OPS["random_drop_all_image"] = random_drop_all_image
AUGMENT_OPS["random_mask"] = random_mask


def augment_image(
    image: tf.Tensor, seed: tf.Tensor, **augment_kwargs
) -> tf.Tensor:
    """Unified image augmentation function for TensorFlow.

    This function is primarily configured through `augment_kwargs`. There must be one kwarg called "augment_order",
    which is a list of strings specifying the augmentation operations to apply and the order in which to apply them. See
    the `AUGMENT_OPS` dictionary above for a list of available operations.

    For each entry in "augment_order", there may be a corresponding kwarg with the same name. The value of this kwarg
    can be a dictionary of kwargs or a sequence of positional args to pass to the corresponding augmentation operation.
    This additional kwarg is required for all operations that take additional arguments other than the image and random
    seed. For example, the "random_resized_crop" operation requires a "scale" and "ratio" argument that can be specified
    either positionally or by name. "random_flip_left_right", on the other hand, does not take any additional arguments
    and so does not require an additional kwarg to configure it.

    Here is an example config:

    ```
    augment_kwargs = {
        "augment_order": ["random_resized_crop", "random_brightness", "random_contrast", "random_flip_left_right"],
        "random_resized_crop": {
            "scale": [0.8, 1.0],
            "ratio": [3/4, 4/3],
        },
        "random_brightness": [0.1],
        "random_contrast": [0.9, 1.1],
    ```

    Args:
        image: A `Tensor` of shape [height, width, channels] with the image. May be uint8 or float32 with values in [0, 255].
        seed (optional): A `Tensor` of shape [2] with the seed for the random number generator.
        **augment_kwargs: Keyword arguments for the augmentation operations. The order of operations is determined by
            the "augment_order" keyword argument.  Other keyword arguments are passed to the corresponding augmentation
            operation. See above for a list of operations.
    """
    if "augment_order" not in augment_kwargs:
        raise ValueError("augment_kwargs must contain an 'augment_order' key.")

    # convert to float at the beginning to avoid each op converting back and
    # forth between uint8 and float32 internally
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)

    assert seed is not None, "seed is required in augment_image"
    # if seed is None:
    #     seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)

    for op in augment_kwargs["augment_order"]:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        if op in augment_kwargs:
            if hasattr(augment_kwargs[op], "items"):
                image = AUGMENT_OPS[op](image, seed=seed, **augment_kwargs[op])
            else:
                image = AUGMENT_OPS[op](image, seed=seed, *augment_kwargs[op])
        else:
            image = AUGMENT_OPS[op](image, seed=seed)
        # float images are expected to be in [0, 1]
        image = tf.clip_by_value(image, 0, 1)

    # convert back to original dtype and scale
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    return image

# ruff: noqa: B023
def augment(obs: Dict, seed: tf.Tensor, augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images."""
    image_names = [key[6:] for key in obs if key.startswith("image_")]
    depth_names = [key[6:] for key in obs if key.startswith("depth_")]
    # assert image_names == depth_names, "image and depth names must match"

    # "augment_order" is required in augment_kwargs, so if it's there, we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the user has passed in a mapping from image
    # name to augmentation dict)
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            print(f"[Warning]: image topic {name} is not in augment_kwargs ({list(augment_kwargs.keys())})")
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],  # skip padding images
        )
    
    for i, name in enumerate(depth_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting depth_{name} with kwargs {kwargs}")
        depth = obs[f"depth_{name}"]
        intrinsics = obs.get(f"intrinsics_{name}", tf.zeros([3, 3]))
        seed = seed + i
        # in `augment_image`, the true seed is generated by seed
        true_seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        for op in kwargs["augment_order"]:
            if not obs["pad_mask_dict"][f"depth_{name}"]:
                continue
            # depth images serve as gt, so we only need to match them with the corresponding image
            if op == "random_rotation":
                # depth = tf.cond(
                #     obs["pad_mask_dict"][f"depth_{name}"],
                #     lambda: random_rotation(
                #         depth, angle_range=kwargs[op][0], seed=true_seed, interpolation='nearest'),
                #     lambda: depth,
                # ) # note the kwargs[op][0] is to be compatible with *kwargs[op]
                res = random_rotation(
                    depth, angle_range=kwargs[op][0], seed=true_seed, interpolation='nearest', intrinsics=intrinsics)
                depth = res["image"]
                intrinsics = res["intrinsics"]
                
            elif op == "random_resized_crop":
                # depth = tf.cond(
                #     obs["pad_mask_dict"][f"depth_{name}"],
                #     lambda: random_resized_crop_depth(
                #         depth, scale=kwargs[op]["scale"], ratio=kwargs[op]["ratio"], seed=true_seed),
                #     lambda: depth,
                # )
                res = random_resized_crop_depth(
                    depth, scale=kwargs[op]["scale"], ratio=kwargs[op]["ratio"], seed=true_seed, intrinsics=intrinsics)
                depth = res["image"]
                intrinsics = res["intrinsics"]
                
            elif op == "random_drop_all_image":
                # depth = tf.cond(
                #     obs["pad_mask_dict"][f"depth_{name}"],
                #     lambda: random_drop_all_image(depth, drop_prob=kwargs[op][0], seed=true_seed),
                #     lambda: depth,
                # )
                depth = random_drop_all_depth(depth, drop_prob=kwargs[op][0], seed=true_seed)
            elif op == "random_mask":
                # depth = tf.cond(
                #     obs["pad_mask_dict"][f"depth_{name}"],
                #     lambda: random_mask(depth, mask_scale=kwargs[op][0], seed=true_seed),
                #     lambda: depth,
                # )
                depth = random_mask(depth, mask_scale=kwargs[op][0], seed=true_seed)
            else:
                continue
        obs[f"depth_{name}"] = depth
        obs[f"intrinsics_{name}"] = intrinsics

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                "padding depth images, which may cause errors if you mix padding and non-padding images."
            )
        depth = obs[f"depth_{name}"]

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32)
            else:
                depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.uint16)
                depth = tf.cast(depth, tf.float32) / 1000.0 # depth images are in mm
        elif depth.dtype != tf.float32:
            raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}")

        if name in depth_resize_size:
            # depth images should be resized using nearest neighbor, not bilinear
            # depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])
            orig_h, orig_w = tf.shape(depth)[0], tf.shape(depth)[1]
            depth = tf.image.resize(depth, depth_resize_size[name], method="nearest")
            new_h, new_w = tf.shape(depth)[0], tf.shape(depth)[1]
            # deal with potential intrinsics
            if f"intrinsics_{name}" in obs:
                intrinsics = obs[f"intrinsics_{name}"]
                scale_matrix = tf.convert_to_tensor([
                    [new_w/orig_w, 0, 0], 
                    [0, new_h/orig_h, 0], 
                    [0, 0, 1]
                ], dtype=tf.float32)
                intrinsics = tf.matmul(scale_matrix, intrinsics)
                obs[f"intrinsics_{name}"] = intrinsics

        obs[f"depth_{name}"] = depth

    return obs

