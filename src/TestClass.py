import torch
PATH = "/Users/pingwin/Documents/GitHub/AI_Project/savedModel/trainedModel.pth"

model = torch.load(PATH,map_location=torch.device('cpu'))

for parameter in model.parameters() :
    parameter.requires_grad = False

model.eval()


