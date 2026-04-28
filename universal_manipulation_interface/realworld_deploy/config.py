# Network
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007

SOCKET_TIMEOUT_SEC = 10.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 2


# Policy inference


# POLICY_CHECKPOINT_PATH = (
#     "/home/sunpeng/sp/umi_project/universal_manipulation_interface/"
#     "data/outputs/2026.04.20/16.30.06_train_diffusion_unet_timm_picknplace/checkpoints"
# ) ## fisheye + crop 30 epoch
# CROP = True


# POLICY_CHECKPOINT_PATH = (
#     "/home/sunpeng/sp/umi_project/universal_manipulation_interface/"
#     "data/outputs/2026.04.27/11.03.05_train_diffusion_unet_timm_picknplace/checkpoints"
# ) ## RGB + no crop 10 epoch
# CROP = False

# POLICY_CHECKPOINT_PATH = (
#     "/home/sunpeng/sp/umi_project/universal_manipulation_interface/"
#     "data/outputs/2026.04.27/14.26.30_train_diffusion_unet_timm_picknplace/checkpoints"
# ) ## RGB + crop 30 epoch
# CROP = True



POLICY_CHECKPOINT_PATH = (
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.28/08.42.57_train_diffusion_unet_timm_picknplace/checkpoints"
) ## fisheye + no crop 40 epoch
CROP = False



DEFAULT_POLICY_ARM = "arm_l"

# Logging / saving
VERBOSE = True     ## Whether to print verbose messages.
PRINT = True     ## Whether to print payload records.
PICT_SAVE = True ## Whether to save payload records.

# policy inference
ACTION_CHUNK_HORIZON = 16 ## Keep the first N actions from the predicted action chunk
