import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import gradio as gr

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

feature_layers = [0,5,10,19,28]

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.feature_layers = feature_layers
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0,ceil_mode=False)
    def forward(self,x):
        style_features = []
        for i,layer in enumerate(self.model.features[:29]):
            if isinstance(layer,nn.MaxPool2d):
                x = self.avg_pool(x)
                continue
            x = layer(x)
            if i in self.feature_layers:
                style_features.append(x)
            if i == 23:
                content_features = x
            
        return style_features,content_features
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_merger(content, style,beta=10,device=device):
    size = 300
    alpha = 1
    beta *= 1000
    content = Image.fromarray(content)
    style = Image.fromarray(style)
    t = transforms.Compose(
        [
            transforms.Resize((size,size)),
            transforms.ToTensor(),
        ]
    )
    style = t(style).unsqueeze(0).to(device)
    content = t(content).unsqueeze(0).to(device)
    generated = content.clone().to(device).requires_grad_(True)
    generator = StyleTransfer().to(device).eval()
    opt = torch.optim.Adam([generated],lr=0.06)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)  # Learning rate scheduler
    num_epochs = 30 if device != "cuda" else 100
    style_features,_ = generator(style)
    _,content_features = generator(content)
    loop = tqdm(range(num_epochs),leave=False)
    for i in loop:
        content_loss = 0
        style_loss = 0
        generated_style_features,generated_content_features = generator(generated)
        content_loss = 0.5 * torch.mean((content_features - generated_content_features) ** 2)
        for style_feature,generated_style_feature in zip(style_features,generated_style_features):
            b,c,h,w = style_feature.shape
            s1 = style_feature.view(c,h*w) @ style_feature.view(c,h*w).T
            s2 = generated_style_feature.view(c,h*w) @ generated_style_feature.view(c,h*w).T

            layer_style_loss = torch.mean((s2 - s1)**2)/(4 *(c) * (h*w))
            style_loss += layer_style_loss
        total_loss = alpha * content_loss + beta * style_loss
        loop.set_postfix(loss=total_loss.item())
        opt.zero_grad()
        total_loss.backward(retain_graph=True)
        opt.step()
        scheduler.step()
        if total_loss < 200 and device!='cuda':
            break
    print(total_loss.item())
    img = np.array(generated.cpu().detach().squeeze(0).permute(1,2,0)) 
    img = np.clip(img,0,1) * 255
    img = Image.fromarray(img.astype(np.uint8))
    return img

iface = gr.Interface(
    fn=image_merger,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Image(label="Style Image"),
        gr.Slider(label="Style strength", minimum=10, maximum=100, step=10),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Neural Style Transfer",
    description="Upload your desired input image and style image. Adjust the 'Style strength' slider to control the intensity of the style transfer. The generated image will showcase your input content with the stylistic elements of the chosen style image. Generation can take upto two minutes",
)

iface.launch()