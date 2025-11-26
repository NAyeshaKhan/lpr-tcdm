import torch
import os
from text_recognition.recognizer_init import RecognizerBuilder

def build_aster_model():
    """
    Build ASTER model architecture only (no checkpoint loading)
    """
    # Adjust these parameters based on your original Aster_init
    imgH = 32
    nc = 1          # Number of input channels
    nclass = 37     # Number of character classes
    nh = 256        # Size of LSTM hidden state

    model = RecognizerBuilder(imgH=imgH, nc=nc, nclass=nclass, nh=nh)
    return model

def check_aster_checkpoint(model_path):
    if not os.path.exists(model_path):
        print(f"Checkpoint file does not exist: {model_path}")
        return

    # Load checkpoint manually
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())

    # Build model architecture only
    ASTER = build_aster_model()

    # Load checkpoint state_dict into model
    try:
        missing_keys, unexpected_keys = ASTER.load_state_dict(checkpoint['state_dict'], strict=False)
        print("ASTER model loaded successfully!")
        if missing_keys:
            print("\nMissing keys in model:", missing_keys)
        if unexpected_keys:
            print("\nUnexpected keys in checkpoint:", unexpected_keys)
    except RuntimeError as e:
        print("Error loading state_dict into ASTER model:")
        print(e)

if __name__ == "__main__":
    model_path = "./text_recognition/ckpt/aster_demo.pth.tar"  # Replace with your checkpoint path
    check_aster_checkpoint(model_path)
