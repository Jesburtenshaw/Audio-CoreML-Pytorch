import torch
import os
import glob
import coremltools as ct
import numpy as np

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
    net_g.eval()#.to("device")
    net_g.remove_weight_norm()
    return net_g, cpt

def exportVoiceModel(modelpath):
    outputpath = os.path.splitext(modelpath)[0]+'mps.ts'
    print("Input: " + modelpath)
    print("Output: " + outputpath)
    device = torch.device("mps")
    model, cpt = get_synthesizer(modelpath, device)
    assert isinstance(cpt, dict)

    scriptedmodel = torch.jit.script(model)
    scriptedmodel = scriptedmodel.to(device)
    scriptedmodel.save(outputpath)

#exportVoiceModel("./Models/Ajuna/Anjuna_2.pth")
# for root, dirs, files in os.walk("./Models"):
#     for file in files:
#         if file.endswith(".pth"):
#             modelfile = os.path.join(root, file)
#             exportVoiceModel(modelfile)

def main():
    model_path = "./LISA.pth"
    model, cpt = get_synthesizer(model_path)
    print("Model loaded")

if __name__ == "__main__":
    main()