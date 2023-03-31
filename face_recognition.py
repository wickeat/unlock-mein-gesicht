import torch
import torch.nn as nn
import torchvision.models as models

class FaceNet(nn.Module):
    def __init__(self, embedding_size):
        super(FaceNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        x = self.model(x)
        return x

model = FaceNet(embedding_size=128)
checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    embedding = model(image).numpy()
