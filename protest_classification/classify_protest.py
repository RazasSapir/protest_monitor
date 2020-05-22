import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from protest_classification.utils import ProtestEvalDir, ProtestEvalImage, modified_resnet50


NUM_WORKERS = 4
DIR_BATCH_SIZE = 16
COLUMNS = ["Protest", "Violence Rate", "Sign", "Photo", "Fire", "Police",
           "Children", "Over 20 people", "Over 100 people", "Flag", "Night", "Shouting"]


def eval_one_dir(model, img_dir):
    dataset = ProtestEvalDir(img_dir=img_dir)
    data_loader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=DIR_BATCH_SIZE)
    output_dicts = []
    for sample in data_loader:
        img_path, input_data = sample['imgpath'], sample['image']
        input_var = Variable(input_data)
        output = model(input_var).cpu().data.numpy().reshape(12)
        output_dicts.append(output_to_dict(output))

    return output_dicts


def eval_image(model, img_path):
    dataset = ProtestEvalImage(img_path)
    data_loader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=1)
    for single_image in data_loader:
        input_data = single_image['image']
        input_var = Variable(input_data)
        output = model(input_var).cpu().data.numpy().reshape(12)
        output_dict = output_to_dict(output)
        return output_dict


def output_to_dict(outputs):
    output_dict = {}
    for i, col in enumerate(COLUMNS):
        output_dict[col] = round(float(outputs[i]) * 100, 2)  # float to percentage
    return output_dict


def get_model(model_path):
    model = modified_resnet50()
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f,  map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    return model


def main():
    pass


if __name__ == "__main__":
    main()
