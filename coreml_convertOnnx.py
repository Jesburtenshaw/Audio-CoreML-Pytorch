#!/usr/bin/env python

# This script creates a RVC model (from onnx_models_onnx), trace it and convert it to CoreML
# To be able to trace without 'intimplicit' op issue, you need to patch Retrieval-based-Voice-Conversion-WebUI sources
# Remove the @torch.jit.script decorator
# from fused_add_tanh_sigmoid_multiply function in infer/lib_src/infer_pack/commons.py

# NOTE TO MORNING MICHAEL
# unpatch RVC, try exporting onnx model only again

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
vec_channels = 768

test_phone = torch.rand(1, 200, vec_channels).float()
test_phone_lengths = torch.tensor([200]).long()
test_pitch = torch.randint(size=(1, 200), low=5, high=255)
test_pitchf = torch.rand(1, 200).float()
test_ds = torch.LongTensor([0])

from infer.lib_src.infer_pack.models import SynthesizerTrnMs768NSFsid

model = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False,
    version=cpt.get("version", "v1"))
model.load_state_dict(cpt["weight"], strict=False)
input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds"]
output_names = ["audio"]
model.eval().to(device)

traced_model = torch.jit.trace(model,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
        ),
        check_trace=True
)

print(test_phone.dtype, test_phone_lengths.dtype,
  test_pitch.dtype, test_pitchf.dtype,
  test_ds.dtype)

mlmodel = ct.converters.convert(traced_model,
    convert_to='mlprogram',
    inputs=[
        ct.TensorType(name='phone', shape=(1, 200, vec_channels), dtype=np.float32),
        ct.TensorType(name='phone_lengths', shape=test_phone_lengths.shape, dtype=np.int64),
        ct.TensorType(name='pitch', shape=test_pitch.shape, dtype=np.int64),
        ct.TensorType(name='pitchf', shape=test_pitchf.shape, dtype=np.float32),
        ct.TensorType(name='ds', shape=test_ds.shape, dtype=np.int64),
        ],
    outputs=[ct.TensorType(name='audio')])

print("Saving CoreML Package")
mlmodel.save(outputPath)

print ("Saving ONNX model")
torch.onnx.export(
        traced_model,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
        ),
        onnxOutputPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )