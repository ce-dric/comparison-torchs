import torch
import time


class VGG19(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.feature = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
            ]
        )

        self.classifier = torch.nn.Sequential(
            *[
                torch.nn.Linear(512 * 7 * 7, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, self.num_classes),
            ]
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x


def main():
    total_train_time = 0.0
    total_valid_time = 0.0

    start_program = time.time()
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available! Training on CPU.")
        return

    batch_size = 64
    epochs = 100

    model = VGG19(10)
    model.to("cuda:0")

    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_total = time.time()
    for epoch in range(epochs):
        start_train = time.time()
        model.train()
        train_loss = 0.0
        for iter in range(10):
            inputs = torch.rand((batch_size, 3, 224, 224)).to("cuda:0")
            labels = torch.rand((batch_size, 10)).to("cuda:0")

            pred = model(inputs)
            loss = criterion(pred, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()

        print(f"Epoch {epoch}: Average Training Loss = {(train_loss / 10)}")
        torch.cuda.synchronize()
        end_train = time.time()
        total_train_time += end_train - start_train

        start_val = time.time()
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for iter in range(10):
                val_inputs = torch.rand((batch_size, 3, 224, 224)).to("cuda:0")
                val_labels = torch.rand((batch_size, 10)).to("cuda:0")
                val_pred = model(val_inputs)
                val_loss = criterion(val_pred, val_labels)
                valid_loss += val_loss.item()

        print(f"Epoch {epoch}: Average Validation Loss = {(valid_loss / 10)}")
        torch.cuda.synchronize()
        end_val = time.time()
        total_valid_time += end_val - start_val

    torch.cuda.synchronize()
    end_total = time.time()
    end_program = time.time()

    print("Results")

    print(
        f"Average Train Time per Epoch: {total_train_time / epochs} seconds."
    )
    print(
        f"Average Valid Time per Epoch: {total_valid_time / epochs} seconds."
    )
    print(f"Total Train Time: {total_train_time} seconds.")
    print(f"Total Valid Time: {total_valid_time} seconds.")
    print(f"Total Time: { end_total - start_total} seconds.")
    print(f"Total Program Time: { end_program - start_program} seconds.")


if __name__ == "__main__":
    main()
