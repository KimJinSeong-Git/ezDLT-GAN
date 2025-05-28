import torch
from torch.utils.data import DataLoader
import yaml, os

from features import model, trainer, dataset, classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using '{device}' now.")

def load_config(root_config="default.yaml"):
    with open(root_config, "r") as file:
        config = yaml.safe_load(file)

    config["save_dir"] = f'./results/{config["prefix"]} {config["model"]["type"]}_{config["dataset"]["name"]}_z{config["model"]["z_size"]}_f{config["model"]["nb_feat_maps"]}'

    return config

def train_model(config):
    # load GAN model setting
    model_type = config["model"]["type"]
    z_size = config["model"]["z_size"]
    nb_feat_maps = config["model"]["nb_feat_maps"]

    # load dataset setting
    root_dataset = config["dataset"]["root_dataset"]
    image_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]

    train_dataset = dataset.TrainDataset(root_dataset, image_size=(image_size[1], image_size[2]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # load training setting
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # set training environment
    if model_type == "WGAN":
        gan_model = model.WGAN(image_size[1], z_size, nb_feat_maps).to(device)

        gan_trainer = trainer.WGAN_Traininer(device, gan_model, train_loader, image_size[1], z_size, learning_rate, save_dir)
        gan_trainer.run(epochs)
    elif model_type == "InversionGAN":
        root_target_model = config["model"]["root_target_model"]
        type_target_model = config["model"]["type_target_model"]
        hidden_size = config["model"]["hidden_size"]
        layer_depth = config["model"]["layer_depth"]
        nb_classes = config["model"]["nb_classes"]
        
        target_model = classifier.CNN_Model(type_target_model, input_size=image_size, hidden_size=hidden_size, layer_depth=layer_depth, nb_classes=nb_classes).to(device)
        target_model.load_state_dict(torch.load(root_target_model))
        gan_model = model.InversionGAN(image_size[1], z_size, device, nb_feat_maps, nb_classes)

        gan_trainer = trainer.InversionGAN_Trainer(device, gan_model, target_model, train_loader, nb_classes, z_size, learning_rate, save_dir)
        gan_trainer.run(epochs, train_dataset, batch_size)
    else:
        print("WHAT???!?!??@#?!@?#?!@")
    
if __name__ == "__main__":
    is_custom_yaml = False

    if is_custom_yaml:
        dir_custom_yaml = './yamls'
        list_custom_yaml = os.listdir(dir_custom_yaml)
        for custom_yaml in list_custom_yaml:
            config = load_config(f"{dir_custom_yaml}/{custom_yaml}")
            train_model(config)
            torch.cuda.empty_cache() 
    else:
        config = load_config("setting.yaml")
        train_model(config)