#importing libraries
import torch 
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

#using VGG-19 pretrained model
model=models.vgg19(pretrained=True).features

#Model
class VGG(nn.Module):
  def __init__(self):
    super(VGG,self).__init__()
    #choosing only the layers mentioned here in the list (the count can be obtained by printing model.features)
    self.chosen_features=['0','5','10','19','28']
    self.model=models.vgg19(pretrained=True).features[:29]
  def forward(self,x):
    features=[]
    for layer_num,layer in enumerate(self.model):
      x=layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)
      return features

#loading the image and adding one more dimension
def load_image(image_name):
  image=Image.open(image_name)
  image=loader(image).unsqueeze(0)
  return image.to(device)


device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
image_size=356

#Data manipulation
loader=transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
]
)
original_img=load_image('/content/drive/MyDrive/anne.jpg')
style_img=load_image('/content/drive/MyDrive/van gogh.jpg')



#the original generated image will be the same as original image
generated=original_img.clone().requires_grad_(True)
model=VGG().to(device).eval()

#parameters
total_steps=6000
learning_rate=0.01
alpha=1
beta=0.01

optimizer=optim.Adam([generated],lr=learning_rate)
#the model is trained,loss is implemented and backpropagation is done
for step in range(total_steps):
  generated_features=model(generated)
  original_img_features=model(original_img)
  style_features=model(style_img)
  style_loss=original_loss=0
  for gen_feature,orig_feature,style_feature in zip(
      generated_features,original_img_features,style_features
  ):
      batch_size,channel,height,width=gen_feature.shape
      original_loss+=torch.mean(gen_feature-orig_feature
**2)
      G=gen_feature.view(channel,height*width).mm(
          gen_feature.view(channel,height*width).t()
      )
      A=style_feature.view(channel,height*width).mm(
          style_feature.view(channel,height*width).t()
      )
      style_loss+=torch.mean((G-A)**2)
  total_loss=alpha*original_loss+beta*style_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()
  if step%200==0:
    print(total_loss)
    save_image(generated,'/content/drive/MyDrive/generated.png')

