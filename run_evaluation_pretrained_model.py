from config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from utils import assign_gpu
from moudle_test import  do_test
import torch


def test(dataset_name, feature_path,  pretrained_model_path, config_file='config/config_pretrained.json'):
    args = get_config_regression(dataset_name, config_file)
    args.featurePath = feature_path
    args.device = assign_gpu([0])

    dataloader = MMDataLoader(args, 1)
    model = AMIO(args).to(args['device'])

    #load pretrained_model
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(args.device)

    do_test(args, model, dataloader["test"])


if __name__ == '__main__':
    dataset_name = "attractiveness"
    feature_path = r"data/Salesperson_Attractiveness_data1993.pkl"
    pretrained_model_path = r"pretrained_model/tfn_pretrained.pth"

    test(dataset_name, feature_path,  pretrained_model_path)
