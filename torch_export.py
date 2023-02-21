import torch
from isegm.inference import utils

# Load the Torch model
model = utils.load_is_model("weights/focalclick_hr18ss1_cclvs.pth", torch.device(f'cuda:{0}'), cpu_dist_maps=True)
dummy_data_1 = torch.randn(1, 4, 256, 256, device='cuda')
dummy_data_2 = torch.randn(1, 2, 3, device='cuda')
# Create a TensorFlow SavedModel
model.eval()
traced_script_module = torch.jit.trace(model, (dummy_data_1, dummy_data_2))
traced_script_module.save("weights/model.pt")