from torch import nn


class Model(nn.Module):
    def __init__(self, original_model, num_classes):
        super(Model, self).__init__()
        self.features = original_model.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )



        self.modelName = 'alexnet'

        # for p in self.features.parameters() :
        #     p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
