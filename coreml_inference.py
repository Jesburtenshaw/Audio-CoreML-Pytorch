#!/usr/bin/env python

import torch
import coremltools as ct

model = ct.models.MLModel('rvc.mlpackage')
print(model.input_description)
print(model.output_description)

test_phone = torch.rand(1, 200, 768)
test_phone_lengths = torch.tensor([200.0])
test_pitch = torch.rand(1, 200)
test_pitchf = torch.rand(1, 200)
test_ds = torch.tensor([0.0])
test_rnd = torch.rand(1, 192, 200)

dict = {
  "phone":test_phone,
  "phone_lengths":test_phone_lengths,
  "pitch":test_pitch,
  "pitchf":test_pitchf,
  "ds":test_ds,
  "rnd":test_rnd,
}

output = model.predict(dict)

audio = output['audio']
print("Audio shape:", audio.shape)
