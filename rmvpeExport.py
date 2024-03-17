import torch

from infer.lib_src.rmvpe import RMVPE
from infer.lib_src.jit.get_rmvpe import get_rmvpe

print("Loading rmvpe model")
#model = get_rmvpe("./Liza/rmvpe.pt", device=torch.device("cpu"))

model=RMVPE("assets/rmvpe/rmvpe.pt", is_half=False, device="mps", use_jit=True)
#jitmodel = model.get_jit_model()

#model.eval()
rmvpescript = torch.jit.script(model)

rmvpescript.save("./Liza/rmvpe_script.ts")