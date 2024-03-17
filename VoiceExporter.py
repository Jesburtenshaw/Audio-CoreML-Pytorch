import torch
import os
import glob
import coremltools as ct
import numpy as np
from scipy.io.wavfile import write


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.lib.infer_pack.models import (
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
    phone_lengths = state_dict['phone_length'].long()
    pitch = state_dict['pitch'].long()
    pitchf = state_dict['pitchf'].float()
    ds = state_dict['ds'].long()

    example_output_file = "./assets/output.pt"
    example_output = torch.jit.load(example_output_file)
    state_dict = example_output.state_dict()
    audio = state_dict['audio']
    print(audio.shape)
    audio_np = audio.numpy()

    # Reshape the array to 1D
    audio_np = np.reshape(audio_np, -1)

    # Play the audio
    # sd.play(audio_np, samplerate=16000)  #
    # sd.wait()
    # write('output.wav', 16000, audio_np)

    model_path = "lisa/LISA.pth"

    onnx_output_path = "./lisa/LISA.onnx"
    coreml_output_path = "./lisa/LISA.mlpackage"

    device = torch.device("cpu")
    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)


    res = model(phone, phone_lengths, pitch, pitchf, ds)
    result_audio = res[0].detach().numpy()
    result_audio = np.reshape(result_audio, -1)
    write('result.wav', 16000, result_audio)

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

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "skip", "return"]
    output = traced_model(phone, phone_lengths, pitch, pitchf, ds)
    print("Saving ONNX model")
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
    import onnx
    import onnxruntime

    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)

    session = onnxruntime.InferenceSession(onnx_output_path)
    input_names = [input.name for input in session.get_inputs()]
    print("Input names: ", input_names)

    input_data = {
        'phone': phone.numpy().astype(np.float32),
        'phone_lengths': phone_lengths.numpy().astype(np.int32),
        'pitch': pitch.numpy().astype(np.int32),
        'pitchf': pitchf.numpy().astype(np.float32),
        'ds': ds.numpy().astype(np.int32),
    }

    output = session.run(None, input_data)

    onnx_audio = output[0]
    onnx_audio = np.reshape(onnx_audio, -1)
    write('onnx_audio.wav', 16000, onnx_audio)

    distance_each_channel = np.abs(audio_np - onnx_audio)
    print("Max distance: ", np.max(distance_each_channel))



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
        compute_precision=ct.precision.FLOAT16,
    )


    print("Saving CoreML Package")
    mlmodel.save(coreml_output_path)

    ml_model = ct.models.MLModel(coreml_output_path)
    output = ml_model.predict(input_data)
    print(output)




if __name__ == "__main__":
    main()
