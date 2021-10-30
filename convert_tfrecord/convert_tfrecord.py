import os
import json
import argparse
import functools

import numpy as np
import tensorflow as tf
import tfrecord_reading_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, "metadata.json")) as f:
        metadata = json.load(f)

    for split in ["train", "valid", "test"]:
        ds = tf.data.TFRecordDataset([os.path.join(args.data_path, f"{split}.tfrecord")])
        ds = ds.map(functools.partial(
            tfrecord_reading_utils.parse_serialized_simulation_example, metadata=metadata))
        data_shape = {}
        file_offset = {}
        for sample in ds.as_numpy_iterator():
            context, parsed_features = sample
            data = {"particle_type": context["particle_type"], **parsed_features}
            shape = {}
            for key, value in data.items():
                filename = os.path.join(args.data_path, split + "_" + key + ".dat")
                offset = file_offset.get(key, 0)
                if key == "particle_type":
                    assert value.dtype == np.int64, value.dtype
                else:
                    assert value.dtype == np.float32, value.dtype
                mode = "r+" if os.path.exists(filename) else "w+"
                array = np.memmap(filename, dtype=value.dtype, mode=mode, offset=offset * value.dtype.itemsize, shape=value.shape)
                array[:] = value
                shape[key] = {"offset": offset, "shape": value.shape}
                file_offset[key] = offset + value.size
            data_shape[int(context["key"])] = shape
        with open(os.path.join(args.data_path, split + "_offset.json"), "w") as f:
            json.dump(data_shape, f, indent=2)


if __name__ == "__main__":
    main()
