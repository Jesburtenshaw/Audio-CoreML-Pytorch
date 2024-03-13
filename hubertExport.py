##DOESN'T WORK
##USE HUBERTEXPORT2 and read the instructions

import torch

from infer.lib.jit.get_hubert import get_hubert_model

device = torch.device("cpu")

print("Loading hubert model")
model = get_hubert_model("./assets/hubert/hubert_base.pt", device=device)

feats = torch.randn(1, 16000, dtype=torch.float32)
padding_mask = torch.zeros(feats.shape, dtype=torch.bool) # constant

model.eval()
# hubertscript = torch.jit.script(model)
hubertscript = torch.jit.trace(model,
                               (feats.to(device),
                                padding_mask.to(device)
                                ),
                                check_trace=True
                                )

hubertscript.save("./Liza/hubert.ts")