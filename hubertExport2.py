import torch
from fairseq import checkpoint_utils
import coremltools as ct
import numpy as np
#TO USE
#Uncomment the original code to setup the outputs
#Use fairseq.zip to replace fairseq/models/hubert/hubert.py and fairseq/models/wav2vec/utils.py

#if you are trying to trace with mps to verify mps success, see hubertmpsfix.png for the patch

models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    ["assets/hubert/hubert_base.pt"],
    suffix="",
)

device = torch.device("mps")

model = models[0].to(device)
model = model.eval()

# Original code, saving it for comparison later
# feats = torch.randn(1, 16160, dtype=torch.float32)
# padding_mask = torch.zeros(feats.shape, dtype=torch.bool) # constant
# inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 12}
# torch.save(feats, "hubertassets/feats.pt")
# with torch.no_grad():
#     logits = model.extract_features(**inputs)
#     torch.save(logits[0], "hubertassets/logit.pt")
#     y = model.final_proj(logits[0])
#     torch.save(y, "hubertassets/y.pt")

feats = torch.load("hubertassets/feats.pt").to(device)
logit = torch.load("hubertassets/logit.pt").to(device)
y = torch.load("hubertassets/y.pt").to(device)

# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

# merge (extract_features + final_proj)
model_ts = torch.jit.trace(model, feats)
model_ts.save("hubertassets/hubert_base_ts.pt")
y1 = model_ts(feats)
print(torch.max(torch.abs(y - y1)))

# mlmodel = ct.converters.convert(model_ts,
#     convert_to='mlprogram',
#     inputs=[
#         ct.TensorType(name='feats', shape=feats.shape, dtype=np.float32),
#         ],
#     outputs=[ct.TensorType(name='audio')])
#
# print("Saving CoreML Package")
# mlmodel.save("hubertassets/hubert.mlpackage")

print("Saving ONNX model")
torch.onnx.export(
        model_ts,
        feats,
        "hubertassets/hubert.onnx",
        do_constant_folding=True,
        dynamic_axes={
            "feats": [1],
        },
        opset_version=16,
        verbose=False,
        input_names=["feats"],
        output_names=["output"],
    )