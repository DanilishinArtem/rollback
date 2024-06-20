from datasets import load_dataset
from datasets import load_from_disk
from torchvision import transforms


def load_mnist_dataset():
    dataset = load_from_disk("/home/adanilishin/rollback/mnist/mnist_dataset")
    # dataset = load_dataset("mnist")
    # dataset.save_to_disk("./mnist_dataset")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def transform_dataset(batch):
        batch["image"] = [transform(image.convert("L")) for image in batch["image"]]
        return batch

    dataset = dataset.with_transform(transform_dataset)
    return dataset