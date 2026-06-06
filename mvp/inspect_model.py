import os, sys

MODEL_PATH = "mataas.h5"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found")
    sys.exit(1)

ext = os.path.splitext(MODEL_PATH)[1].lower()

if ext in (".h5",):
    import tensorflow as tf

    # Keras 3 dropped the 'groups' arg from DepthwiseConv2D; strip it so
    # models saved with Keras 2 / TF2.x still load.
    class _DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"DepthwiseConv2D": _DepthwiseConv2D},
        compile=False,
    )
    print(f"Model type: Keras .h5")
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print()
    print("Layer summary:")
    model.summary()
    h, w, c = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    n_out = model.output_shape[-1]
    print()
    print(f"SUMMARY: IMAGE model — expects {w}x{h} image with {c} channel(s), {n_out} output classes")

elif ext == ".tflite":
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    print("input_details:", interp.get_input_details())
    print("output_details:", interp.get_output_details())

elif ext in (".p", ".pkl"):
    import pickle
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    if hasattr(model, "n_features_in_"):
        print(f"n_features_in_: {model.n_features_in_}")
    if hasattr(model, "classes_"):
        print(f"classes_: {model.classes_}")
    n = getattr(model, "n_features_in_", "?")
    print(f"\nSUMMARY: LANDMARKS model — expects {n} numbers as input")
