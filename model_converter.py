import os
import pickle
import glob
import time

import torch

import coremltools as ct
import numpy as np
from scipy.io.wavfile import write
import onnxruntime


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.lib_src.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    # tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.forward = net_g.infer
    # ckpt = {}
    # ckpt["config"] = cpt["config"]
    # ckpt["f0"] = if_f0
    # ckpt["version"] = version
    # ckpt["info"] = cpt.get("info", "0epoch")
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval()  # .to("device")
    net_g.remove_weight_norm()
    return net_g, cpt


def exportVoiceModel(modelpath):
    outputpath = os.path.splitext(modelpath)[0] + 'mps.ts'
    print("Input: " + modelpath)
    print("Output: " + outputpath)
    device = torch.device("mps")
    model, cpt = get_synthesizer(modelpath, device)
    assert isinstance(cpt, dict)

    scriptedmodel = torch.jit.script(model)
    scriptedmodel = scriptedmodel.to(device)
    scriptedmodel.save(outputpath)


# exportVoiceModel("./Models/Ajuna/Anjuna_2.pth")
# for root, dirs, files in os.walk("./Models"):
#     for file in files:
#         if file.endswith(".pth"):
#             modelfile = os.path.join(root, file)
#             exportVoiceModel(modelfile)

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
    skip_head = torch.tensor([250], dtype=torch.int32)
    return_length = torch.tensor([31], dtype=torch.int32)

    phone = torch.concat([phone, phone, phone], axis=1)
    pitch = torch.concat([pitch, pitch, pitch], axis=1)
    pitchf = torch.concat([pitchf, pitchf, pitchf], axis=1)

    input_data = {
        'phone': phone.numpy().astype(np.float32),
        'phone_lengths': phone_lengths.numpy().astype(np.int32),
        'pitch': pitch.numpy().astype(np.int32),
        'pitchf': pitchf.numpy().astype(np.float32),
        'ds': ds.numpy().astype(np.int32),
        'skip_head': skip_head.numpy().astype(np.int32),
        'return_length': return_length.numpy().astype(np.int32),

    }

    model_path = "lisa/LISA.pth"
    onnx_output_path = "./lisa/LISA.onnx"
    coreml_output_path = "./lisa/LISA.mlpackage"
    convert = True
    device = torch.device("cpu")
    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)

    res1 = model(phone, phone_lengths, pitch, pitchf, ds, skip_head, return_length)[0].detach().numpy().flatten()
    res2 = model(phone, phone_lengths, pitch, pitchf, ds, skip_head, return_length)[0].detach().numpy().flatten()

    max_dist = np.max(np.abs(res1 - res2))
    print(f"Max distance between two consecutive inferences: {max_dist}")

    result_audio = res1

    write('result_audio.wav', 40000, result_audio)

    if convert:
        traced_model = torch.jit.trace(
            model.eval(),
            (
                phone.to(device, dtype=torch.float32),
                phone_lengths.to(device, dtype=torch.int32),
                pitch.to(device, dtype=torch.int32),
                pitchf.to(device, dtype=torch.float32),
                ds.to(device, dtype=torch.int32),
                skip_head.to(device, dtype=torch.int32),
                return_length.to(device, dtype=torch.int32),
            ),
            check_trace=True
        )
        # saved traced model
        traced_model.save("lisa/LISA_traced.pt")
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "skip_head", "return_length"]
        torch.onnx.export(
            traced_model,
            (
                phone.to(device).float(),
                phone_lengths.to(device, dtype=torch.int32),
                pitch.to(device, dtype=torch.int32),
                pitchf.to(device).float(),
                ds.to(device, dtype=torch.int32),
                skip_head.to(device, dtype=torch.int32),
                return_length.to(device, dtype=torch.int32),
            ),
            onnx_output_path,
            dynamic_axes={
                "phone": [1],
                "pitch": [1],
                "pitchf": [1],
            },
            do_constant_folding=True,
            opset_version=17,
            verbose=False,
            input_names=input_names,
            # output_names=output_names,
        )

        session = onnxruntime.InferenceSession(onnx_output_path)
        input_names = [input.name for input in session.get_inputs()]
        print("Input names: ", input_names)
        onnx_input = {input_names[i]: input_data[input_names[i]] for i in range(len(input_names))}
        output = session.run(None, onnx_input)
        onnx_audio = output[0]
        onnx_audio = np.reshape(onnx_audio, -1)
        write('onnx_audio.wav', 40000, onnx_audio)

        average_distance = np.mean(np.abs(result_audio - onnx_audio))
        print(f"Average distance between the converted and onnx model: {average_distance}")

        mlmodel = ct.converters.convert(
            traced_model,
            convert_to='mlprogram',
            inputs=[
                ct.TensorType(name='phone', shape=phone.shape, dtype=np.float32),
                ct.TensorType(name='phone_lengths', shape=phone_lengths.shape,
                              dtype=np.int32),
                ct.TensorType(name='pitch', shape=pitch.shape, dtype=np.int32),
                ct.TensorType(name='pitchf', shape=pitchf.shape, dtype=np.float32),
                ct.TensorType(name='ds', shape=ds.shape, dtype=np.int32),
                ct.TensorType(name='skip_head', shape=skip_head.shape, dtype=np.int32),
                ct.TensorType(name='return_length', shape=return_length.shape, dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name='audio'),
                ct.TensorType(name='x_mask'),
                ct.TensorType(name='z'),
                ct.TensorType(name='z_p'),
                ct.TensorType(name='m_p'),
                ct.TensorType(name='logs_p'),
            ],
            compute_precision=ct.precision.FLOAT16,
        )

        # range_dim = ct.RangeDim(lower_bound=25, upper_bound=200, default=100)
        # input_shape = ct.Shape(
        #     shape=(
        #         1,
        #         range_dim,
        #         vec_channels
        #     ))
        #
        # mlmodel = ct.converters.convert(
        #     traced_model,
        #     convert_to='mlprogram',
        #     inputs=[
        #         ct.TensorType(name='phone', shape=input_shape, dtype=np.float32),
        #         ct.TensorType(name='phone_lengths', shape=(1,), dtype=np.int64),
        #         ct.TensorType(name='pitch', shape=ct.Shape(shape=(1, range_dim)), dtype=np.int64),
        #         ct.TensorType(name='pitchf', shape=ct.Shape(shape=(1, range_dim,)), dtype=np.float32),
        #         ct.TensorType(name='ds', shape=(1,), dtype=np.int64),
        #     ],
        #     outputs=[
        #         ct.TensorType(name='audio'),
        #         ct.TensorType(name='x_mask'),
        #         ct.TensorType(name='z'),
        #         ct.TensorType(name='z_p'),
        #         ct.TensorType(name='m_p'),
        #         ct.TensorType(name='logs_p'),
        #     ],
        #     compute_precision=ct.precision.FLOAT16,
        # )
        print("Saving CoreML Package")
        mlmodel.save(coreml_output_path)

    ml_model = ct.models.MLModel(coreml_output_path)
    warm_up = 5
    for i in range(warm_up):
        output = ml_model.predict(input_data)
    inf_count = 10
    time_sum = 0
    for i in range(inf_count):
        start = time.time()
        output = ml_model.predict(input_data)
        end = time.time()
        time_sum += end - start
    print(f"Average inference time: {time_sum / inf_count} seconds")
    audio1 = ml_model.predict(input_data)["audio"].astype(np.float32).reshape(-1)
    audio2 = ml_model.predict(input_data)["audio"].astype(np.float32).reshape(-1)
    print("Shape is ", audio1.shape)
    write('output_coreml.wav', 40000, audio1)
    average_distance = np.mean(np.abs(result_audio - audio1))
    print(f"Average distance between the original and coreml model: {average_distance}")
    max_dist = np.max(np.abs(audio1 - audio2))
    print(f"Max distance between two consecutive inferences: {max_dist}")



if __name__ == "__main__":
    # test_decoder()
    # save_decoder_input()
    main()
