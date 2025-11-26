import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericCRNN(nn.Module):
    def __init__(self, cnn_checkpoint_path, num_classes=10, rnn_hidden=256):
        super(NumericCRNN, self).__init__()

        # Load the pretrained CNN backbone
        cnn_ckpt = torch.load(cnn_checkpoint_path, map_location='cpu')
        cnn_state = cnn_ckpt['state_dict'] if 'state_dict' in cnn_ckpt else cnn_ckpt

        # Build CNN matching the checkpoint
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   # conv0
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), # conv1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1), # conv2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1), # conv4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Load weights from checkpoint
        self.cnn.load_state_dict({k.replace("model.", ""): v for k, v in cnn_state.items()}, strict=False)

        # Freeze CNN if desired
        for param in self.cnn.parameters():
            param.requires_grad = False

        # LSTM for sequence modeling
        self.rnn = nn.LSTM(
            input_size=512,   # feature channels from CNN
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Linear classifier for digits 0-9
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        conv_out = self.cnn(x)  # [B, C, H, W]

        # Collapse height dimension
        # Assume CNN reduces height to 1 (or small). Use adaptive pooling if needed.
        conv_out = F.adaptive_avg_pool2d(conv_out, (1, conv_out.size(3)))  # [B, C, 1, W]
        conv_out = conv_out.squeeze(2)  # [B, C, W]
        conv_out = conv_out.permute(0, 2, 1)  # [B, W, C] → RNN expects [B, seq_len, feature]

        rnn_out, _ = self.rnn(conv_out)  # [B, W, 2*hidden]
        out = self.fc(rnn_out)  # [B, W, num_classes]

        preds = out.argmax(dim=2)  # [B, W] max at each timestep
        return preds, out
