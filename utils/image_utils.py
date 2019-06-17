from PIL import Image
import numpy as np

#Debug
import tensorflow as tf
from tensorflow.image import ResizeMethod

def letterbox_image(image, size):
    """ Resize image with unchanged aspect ratio using padding.

    Args:
        image: PIL.Image.Image (Jpeg or PNG)
        size: Tuple (416, 416)
    
    Returns:
        new_image: PIL.Image.Image
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    # new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def letterbox_image_tf_dynamic(image, size, resize_method=ResizeMethod.BILINEAR):
    """ Letterbox image that handles dynamic Tensor type """
    if len(image.get_shape()) == 4:
        ih, iw = tf.shape(image)[1], tf.shape(image)[2]
        images = image
    else:
        ih, iw = tf.shape(image)[0], tf.shape(image)[1]
        images = [image]
    w, h = tf.constant(size[0]), tf.constant(size[1])
    scale = tf.minimum(w / iw, h / ih)
    nw = tf.cast(tf.cast(iw, tf.float64) * scale, tf.int32)
    nh = tf.cast(tf.cast(ih, tf.float64) * scale, tf.int32)
    
    image_tensor = tf.image.resize_images(images, (nh, nw), method=resize_method, align_corners=True)
    
    h_pad = tf.cast((h-nh)//2, tf.int32)
    w_pad = tf.cast((w-nw)//2, tf.int32)
    c_pad = 0
    if len(image_tensor.shape) == 4:
        paddings = [[0,0], [h_pad, h_pad], [w_pad, w_pad], [c_pad, c_pad]]
    else:
        paddings = [[h_pad, h_pad], [w_pad, w_pad], [c_pad, c_pad]]
    
    image_tensor = tf.pad(image_tensor, paddings, constant_values=128. / 255.)
    return image_tensor

    

def letterbox_image_tf_static(image, raw_size, tgt_size, resize_method=ResizeMethod.BILINEAR):
    """ Letterbox image that only handles static shape, but more efficiently."""
    if len(image.shape) == 4:
        images = image
    else:
        images = [image]
    
    iw, ih = raw_size
    w, h = tgt_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    h_pad, w_pad, c_pad = (h - nh) // 2, (w - nw) // 2, 0

    image_tensor = tf.image.resize_images(images, (nh, nw), method=resize_method, align_corners=True)
    paddings = [[0,0], [h_pad, h_pad], [w_pad, w_pad], [c_pad, c_pad]]

    image_tensor = tf.pad(image_tensor, paddings, constant_values=128. / 255.)
    return image_tensor


def image_to_ndarray(image, expand_dims=True):
    """ Convert PIL Image to numpy.ndarray and add batch dimension
    
        Args:
            image: PIL.Image.Image
        
        Returns:
            image_data: numpy.ndarray (1, 416, 416, 3) or (416, 416, 3)

    """
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    if expand_dims == True:
        image_data = np.expand_dims(image_data, 0)
    if image_data.shape[-1] == 4:
        image_data = image_data[...,0:-1]
    return image_data

def ndarray_to_image(image_data):
    if len(image_data.shape) == 4:
        image_data = np.squeeze(image_data, axis=0)
    image_data = (image_data * 255).astype("uint8")
    return Image.fromarray(image_data)

def load_yolov3_image(img_fpath):
    """ Load and resize an image for yolo3. """
    model_image_size = (416, 416)
    image = Image.open(img_fpath)
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def l1_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.sum(diff)

def l0_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.count_nonzero(diff)

def l_inf_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.max(diff)

def main():
    image = Image.open('images/cat.jpg')

    boxed_image = letterbox_image(image, tuple(reversed((416,416))))
    image_data_pil = image_to_ndarray(boxed_image, expand_dims=False)
    x_img_pil = tf.placeholder(tf.float32, shape=(416, 416, 3))

    image_data_tf_dynamic = image_to_ndarray(image, expand_dims=False)
    x_img_tf_large = tf.placeholder(tf.float32, shape=(None,None, 3))
    x_img_tf = letterbox_image_tf_dynamic(x_img_tf_large, (416,416))

    image_data_tf_static = image_to_ndarray(image, expand_dims=False)
    x_img_tf_large_static = tf.placeholder(tf.float32, shape=(1080,1920, 3))
    x_img_tf_static = letterbox_image_tf_static(x_img_tf_large_static, (1920, 1080), (416, 416))
    
    with tf.Session() as sess:
        image_resized_pil = sess.run(x_img_pil, feed_dict={x_img_pil: image_data_pil})
        image_resized_tf = sess.run(x_img_tf, feed_dict={x_img_tf_large: image_data_tf_dynamic})
        image_resized_tf = np.squeeze(image_resized_tf, axis=0)
        image_resized_tf_static = sess.run(x_img_tf_static, feed_dict={x_img_tf_large_static: image_data_tf_static})


        l1 = l1_diff(image_resized_tf, image_resized_tf_static)
        l0 = l0_diff(image_resized_tf, image_resized_tf_static)
        l_inf = l_inf_diff(image_resized_tf, image_resized_tf_static)

        print("l1 %f, l0 %d, l_inf %f" % (l1, l0, l_inf))
        image_pil = ndarray_to_image(image_resized_pil)
        image_tf = ndarray_to_image(image_resized_tf)
        image_tf_static = ndarray_to_image(image_resized_tf_static)


        image_tf.save('tf.png')
        image_pil.save('pil.png')
        image_tf_static.save('tf_static.png')

if __name__ == "__main__":
    main()