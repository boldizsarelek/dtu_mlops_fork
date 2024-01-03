import click
import torch
from model import MyAwesomeModel

from data import mnist


PROJECT_DIR = "/Users/boldizsarelek/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/MLOps/dtu_mlops"

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--checkpoint", default=PROJECT_DIR + "/" + "data/checkpoints/model.pt", help="location to save the model")
def train(lr, epochs, checkpoint):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            #Flatten images
            images = images.view(images.shape[0], -1)
            #print(f"flattened shape: {images.shape}")

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        else:
            print(f"Epoch {epoch} - Training loss: {loss.item()}")

    torch.save(model.state_dict(), checkpoint)


@click.command()
@click.option("--checkpoint", default=PROJECT_DIR + "/" + "data/checkpoints/model.pt", help="location to save the model")
def evaluate(checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(checkpoint)

    # TODO: Implement evaluation logic here
    dict = torch.load(checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(dict)
    _, test_set = mnist()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network: {100 * correct / total}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()