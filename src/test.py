import torch
import torchvision.models as models
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision.transforms import transforms

from src.constants import *
from src.model import Model


def main():
    alexnet = models.alexnet(pretrained=True)
    model = Model(alexnet, 2)
    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(254),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    input_image = Image.open('..' + os.path.sep + 'dataset' + os.path.sep + '0' + os.path.sep + '8863_idx5_x201_y1251_class0.png')
    image_transform = transform(input_image)

    batch = torch.unsqueeze(image_transform, 0).to(DEVICE)

    out = model(batch)

    with open(LABELS_PATH) as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(classes[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:]]

    y = classes[index[0]], percentage[index[0]].item()

    draw = ImageDraw.Draw(input_image)
    font = ImageFont.truetype('arial.ttf', 24)
    draw.text((0, 0), str(y), (0, 0, 0), font=font)
    input_image.save('../recognition.jpg')


if __name__ == '__main__':
    main()
