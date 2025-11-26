import torch

class NumericCRNNWrapper:
    def __init__(self, cnn_checkpoint_path, device='cuda'):
        self.device = device
        from text_recognition.model.crnn_numeric import NumericCRNN
        self.model = NumericCRNN(cnn_checkpoint_path).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, imgs, trim_repeats=True):
        """
        imgs: tensor of shape [B, 1, H, W], float32
        trim_repeats: if True, removes repeated consecutive digits (CTC-like)
        Returns:
            list of strings (one per image)
        """
        imgs = imgs.to(self.device)
        preds, _ = self.model(imgs)  # [B, W]

        digits_list = []
        for pred in preds.cpu().numpy():
            s = []
            last = None
            for d in pred:
                if trim_repeats:
                    if d != last:
                        s.append(str(d))
                        last = d
                else:
                    s.append(str(d))
            digits_list.append("".join(s).lstrip('0'))  # optional: remove leading zeros
        return digits_list
