import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from PIL import ImageFont
from PIL import ImageDraw
from torchvision.datasets import ImageFolder

#
# trainSet = ImageFolder('/Users/pingwin/Documents/GitHub/AI_Project/images/train/')
# testSet = ImageFolder('/Users/pingwin/Documents/GitHub/AI_Project/images/val/')

PATH = "/Users/pingwin/Documents/GitHub/AI_Project/savedModel/trainedModel.pth"
model = torch.load(PATH, map_location=torch.device("cpu"))
checkpoint = model.load_state_dict(model['model_state_dict'])
print(checkpoint)
# print(model['model_state_dict'])

# model.load_state_dict(torch.load(PATH)['model_state_dict'])

# transform = transforms.Compose([
#     transforms.Resize(254),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
#                          )])
#
# input_image = Image.open('/Users/pingwin/Documents/GitHub/AI_Project/dataset/0/8863_idx5_x201_y1251_class0.png')
# image_transform = transform(input_image)
#
# batch = torch.unsqueeze(image_transform, 0)
#
# out = model(batch)
#
# with open('labels.txt') as f :
#     classes = [line.strip() for line in f.readlines()]
#
# _, index = torch.max(out, 1)
#
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(classes[index[0]], percentage[index[0]].item())
#
# _, indices = torch.sort(out, descending=True)
# [(classes[idx], percentage[idx].item()) for idx in indices[0][:]]
#
# y = classes[index[0]], percentage[index[0]].item()
#
# draw = ImageDraw.Draw(input_image)
# font = ImageFont.truetype("arial.ttf", 24)
# draw.text((100, 0), str(y), (0, 0, 0), font=font)
# input_image.save('o2.jpg')
