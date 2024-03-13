import torch
import coremltools as ct
import numpy as np

def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
        SLAISynthesizerTrnMs768NSFsid,
    )

    cpt = torch.load(pth_path, map_location=device)
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
            net_g = SLAISynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
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
    net_g.eval()#.to("device")
    net_g.remove_weight_norm()
    return net_g, cpt

inputPath = "./Ajuna/Anjuna_2.pth"
onnxOutputPath = "./Ajuna/Anjuna_2_exp.onnx"
coreMLoutputPath = "./Ajuna/Anjuna.mlpackage"

device = torch.device("cpu")
model, cpt = get_synthesizer(inputPath, device)
assert isinstance(cpt, dict)

scriptedmodel = torch.jit.script(model)
print(scriptedmodel.graph)
#scriptedmodel.eval()

scrmodel = scriptedmodel.to(device)
scrmodel.eval()

test_phone = torch.rand(1, 100, 768)  # hidden unit
test_phone_lengths = torch.tensor([100]).long()  # hidden unit
test_pitch = torch.randint(size=(1, 100), low=5, high=255)
test_pitchf = torch.rand(1, 100)  # nsf
test_ds = torch.tensor([0]).long()  # hidden unit
test_skip = torch.tensor([0]).long()
test_return = torch.tensor([40000]).long()


#model.eval()
traced_model = torch.jit.trace(model.eval(),
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_skip.to(device),
            test_return.to(device),
        ),
        check_trace=True
)

input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "skip", "return"]
output_names = ["audio"]

print ("Saving ONNX model")
torch.onnx.export(
        traced_model,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_skip.to(device),
            test_return.to(device)
        ),
        onnxOutputPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
        },
        do_constant_folding=True,
        opset_version=17,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

# mlmodel = ct.converters.convert(traced_model,
#     convert_to='mlprogram',
#     inputs=[
#         ct.TensorType(name='phone', shape=test_phone.shape, dtype=np.float32),
#         ct.TensorType(name='phone_lengths', shape=test_phone_lengths.shape, dtype=np.int64),
#         ct.TensorType(name='pitch', shape=test_pitch.shape, dtype=np.int64),
#         ct.TensorType(name='pitchf', shape=test_pitchf.shape, dtype=np.float32),
#         ct.TensorType(name='ds', shape=test_ds.shape, dtype=np.int64),
#         ct.TensorType(name='skip', shape=test_skip.shape, dtype=np.int64),
#         ct.TensorType(name='return', shape=test_return.shape, dtype=np.int64),
#         ],
#     outputs=[
#         ct.TensorType(name='audio'),
#         ]
#                                 )
#
# print("Saving CoreML Package")
# mlmodel.save(coreMLoutputPath)
