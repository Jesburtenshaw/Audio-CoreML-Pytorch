#!/usr/bin/env python

# This script creates a RVC model (from onnx_models_onnx), trace it and convert it to CoreML
# To be able to trace without 'intimplicit' op issue, you need to patch Retrieval-based-Voice-Conversion-WebUI sources
# Remove the @torch.jit.script decorator
# from fused_add_tanh_sigmoid_multiply function in infer/lib_src/infer_pack/commons.py


import torch
import numpy as np
import coremltools as ct
import torch.onnx

inputPath = "./Liza/LISA.pth"
outputPath = "./Liza/LISA.mlpackage"
onnxOutputPath = "./Liza/LISA.onnx"

device = "cpu"

cpt = torch.load(inputPath, map_location=torch.device(device))

cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

test_phone = torch.rand(1, 200, vec_channels).float()
test_phone_lengths = torch.tensor([200]).long()
test_pitchInt = torch.randint(size=(1, 200), low=5, high=255)
test_pitchFloat = torch.rand(1, 200).float()
test_ds = torch.LongTensor([0])

from infer.lib_src.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

model = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False,
    version=cpt.get("version", "v1"))
model.load_state_dict(cpt["weight"], strict=False)
input_names = ["phone", "phone_lengths", "pitchFloat", "pitchInt", "ds"]
output_names = ["audio"]
model.eval()

model.to(device)

traced_model = torch.jit.trace(model,
        (
            test_hubert.to(device),
            test_pitchFloat.to(device),
            test_pitchInt.to(device),
        ),
        check_trace=True
)

print(test_hubert.dtype, test_pitchFloat.dtype, test_pitchInt.dtype)

mlmodel = ct.converters.convert(traced_model,
    convert_to='mlprogram',
    inputs=[
        ct.TensorType(name='hubert', shape=(1, 200, vec_channels), dtype=np.float32),
        ct.TensorType(name='pitchFloat', shape=test_pitchFloat.shape, dtype=np.float32),
        ct.TensorType(name='pitchInt', shape=test_pitchInt.shape, dtype=np.int64)
        ],
    outputs=[ct.TensorType(name='audio')])

print("Saving CoreML Package")
mlmodel.save(outputPath)

print ("Saving ONNX model")
# torch.onnx.export(
#         traced_model,
#         (
#             test_hubert.to(device),
#             test_pitchFloat.to(device),
#             test_pitchInt.to(device)
#         ),
#         onnxOutputPath,
#         dynamic_axes={
#             "hubert": [1],
#             "pitchFloat": [1],
#             "pitchInt": [1],
#         },
#         do_constant_folding=False,
#         opset_version=16,
#         verbose=False,
#         input_names=input_names,
#         output_names=output_names,
#     )