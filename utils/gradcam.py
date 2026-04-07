import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    # ✅ Get base model (Xception)
    base_model = model.get_layer("xception")

    # ✅ Get last conv layer from base model
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # ✅ Create a model that maps input → conv output
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # ✅ Create classifier model (top layers)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input
    for layer in model.layers[1:]:  # skip base_model
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    # ✅ Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()
import cv2
import numpy as np

def overlay_heatmap(img, heatmap, alpha=0.4):
    # Convert PIL image → numpy
    img = np.array(img)

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert to 0–255
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img