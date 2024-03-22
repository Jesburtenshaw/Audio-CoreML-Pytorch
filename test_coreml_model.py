import time

import torch

import coremltools as ct
import numpy as np
from scipy.io.wavfile import write
import onnxruntime


def main():
    test_input_file = "./assets/input.pt"
    test_input = torch.jit.load(test_input_file)
    state_dict = test_input.state_dict()

    # Print the keys and values from the state dictionary
    for key, value in state_dict.items():
        print(f'Key: {key}, shape: {value.shape}')

    phone = state_dict['phone'].float()
    phone_lengths = state_dict['phone_length'].to(dtype=torch.int32)
    pitch = state_dict['pitch'].to(dtype=torch.int32)
    pitchf = state_dict['pitchf'].float()
    ds = state_dict['ds'].to(dtype=torch.int32)

    input_data = {
        'phone': phone.numpy().astype(np.float32),
        'phone_lengths': phone_lengths.numpy().astype(np.int32),
        'pitch': pitch.numpy().astype(np.int32),
        'pitchf': pitchf.numpy().astype(np.float32),
        'ds': ds.numpy().astype(np.int32),
    }

    coreml_model = "./lisa/LISA.mlpackage"

    ml_model = ct.models.MLModel(coreml_model)
    warm_up = 2
    for i in range(warm_up):
        output = ml_model.predict(input_data)
    inf_count = 5
    time_sum = 0
    for i in range(inf_count):
        start = time.time()
        output = ml_model.predict(input_data)
        audio = np.array(output["audio"]).astype(np.float32).reshape(-1)
        end = time.time()
        time_sum += end - start
    print(f"Average inference time: {time_sum / inf_count : .5f} seconds")
    audio = np.array(output["audio"]).astype(np.float32).reshape(-1)
    write('output_coreml.wav', 40000, audio)


if __name__ == "__main__":
    main()
