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

    phone = torch.concat([state_dict['phone'].float(), state_dict['phone'].float()])
    phone_lengths = torch.concat([state_dict['phone_length'].to(dtype=torch.int32), state_dict['phone_length'].to(dtype=torch.int32)])
    pitch = torch.concat([state_dict['pitch'].to(dtype=torch.int32), state_dict['pitch'].to(dtype=torch.int32)])
    pitchf = torch.concat([state_dict['pitchf'].float(), state_dict['pitchf'].float()])
    ds = torch.concat([state_dict['ds'].to(dtype=torch.int32), state_dict['ds'].to(dtype=torch.int32)])

    input_data = {
        'phone': phone.numpy().astype(np.float32),
        'phone_lengths': phone_lengths.numpy().astype(np.int32),
        'pitch': pitch.numpy().astype(np.int32),
        'pitchf': pitchf.numpy().astype(np.float32),
        'ds': ds.numpy().astype(np.int32),
    }

    model_path = "lisa/LISA.pth"
    onnx_output_path = "./lisa/LISA.onnx"
    coreml_output_path = "./lisa/LISA.mlpackage"
    convert = True
    device = torch.device("cpu")
    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)

    res = model(phone, phone_lengths, pitch, pitchf, ds)
    result_audio = res[0].detach().numpy()
    result_audio = np.reshape(result_audio, -1)

    if convert:
        traced_model = torch.jit.trace(
            model.eval(),
            (
                phone.to(device, dtype=torch.float32),
                phone_lengths.to(device, dtype=torch.int32),
                pitch.to(device, dtype=torch.int32),
                pitchf.to(device, dtype=torch.float32),
                ds.to(device, dtype=torch.int32),
            ),
            check_trace=True
        )
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds"]
        torch.onnx.export(
            traced_model,
            (
                phone.to(device).float(),
                phone_lengths.to(device, dtype=torch.int32),
                pitch.to(device, dtype=torch.int32),
                pitchf.to(device).float(),
                ds.to(device, dtype=torch.int32),
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
        output = session.run(None, input_data)
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
        print("Saving CoreML Package")
        mlmodel.save(coreml_output_path)

    ml_model = ct.models.MLModel(coreml_output_path)
    ml_model_config = ml_model.get_spec()
    print(ml_model_config)
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
    audio = output["audio"]
    audio = np.array(audio).astype(np.float32).reshape(-1)
    write('output_coreml.wav', 16000, audio)
    average_distance = np.mean(np.abs(result_audio - audio))
    print(f"Average distance between the original and coreml model: {average_distance}")


def save_decoder_input():
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

    model_path = "lisa/LISA.pth"
    onnx_output_path = "./lisa/LISA.onnx"
    coreml_output_path = "./lisa/LISA.mlpackage"
    convert = True
    device = torch.device("cpu")
    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)

    decoder_output = model.get_up_to_decoder(phone, phone_lengths, pitch, pitchf, ds)
    with open('decoder_input.pkl', 'wb') as f:
        pickle.dump(decoder_output, f)

def test_decoder():
    model_path = "lisa/LISA.pth"
    onnx_output_path = "./lisa/LISA.onnx"
    coreml_output_path = "./lisa/LISA.mlpackage"
    convert = True
    device = torch.device("cpu")
    model, cpt = get_synthesizer(model_path, device)

    decoder = model.dec

    with open('decoder_input.pkl', 'rb') as f:
        decoder_input = pickle.load(f)

    decoder_output = decoder(*decoder_input)
    decoder_output = decoder_output[0].detach().numpy().astype(np.float32)
    decoder_output = np.reshape(decoder_output, -1)
    input_dict = {
        'z': decoder_input[0].detach().numpy().astype(np.float32),
        'nsff': decoder_input[1].detach().numpy().astype(np.float32),
        'g': decoder_input[2].detach().numpy().astype(np.float32),
    }
    decoder_onnx_path = "lisa/decoder.onnx"
    decoder_coreml_path = "lisa/decoder.mlpackage"

    traced_decoder = torch.jit.trace(
        decoder.eval(),
        decoder_input,
        check_trace=True
    )
    input_names = ["z", "nsff", "g"]
    torch.onnx.export(
        traced_decoder,
        decoder_input,
        decoder_onnx_path,
        do_constant_folding=True,
        opset_version=17,
        verbose=False,
        input_names=input_names,
    )

    session = onnxruntime.InferenceSession(decoder_onnx_path)
    input_names = [input.name for input in session.get_inputs()]
    print("Input names: ", input_names)
    output = session.run(None, {inp_name: input_dict[inp_name] for inp_name in input_names})
    onnx_output = output[0]
    onnx_output = np.reshape(onnx_output, -1)

    average_distance = np.mean(np.abs(decoder_output - onnx_output))
    print(f"Average distance between the converted and onnx decoder model: {average_distance}")

    mlmodel = ct.converters.convert(
        traced_decoder,
        convert_to='mlprogram',
        inputs=[
            ct.TensorType(name='z', shape=decoder_input[0].shape, dtype=np.float32),
            ct.TensorType(name='nsff', shape=decoder_input[1].shape, dtype=np.float32),
            ct.TensorType(name='g', shape=decoder_input[2].shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name='audio'),
        ],
        compute_precision=ct.precision.FLOAT32,
    )
    print("Saving CoreML Package")
    mlmodel.save(decoder_coreml_path)

    ml_model = ct.models.MLModel(decoder_coreml_path)
    output = ml_model.predict(input_dict)
    audio = output["audio"]
    audio = np.array(audio).astype(np.float32).reshape(-1)
    write('output_coreml_decoder.wav', 16000, audio)
    average_distance = np.mean(np.abs(onnx_output - audio))
    print(f"Average distance between the onnx and coreml decoder model: {average_distance}")

if __name__ == "__main__":
    # test_decoder()
    # save_decoder_input()
    main()
