import torch.nn as nn
from tqdm import tqdm
from utils import assign_gpu, MetricsTop
import torch

def do_test(args, model, dataloader, mode="test"):
    criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
    metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0

    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            vision = batch_data['vision'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            text = batch_data['text'].to(args.device)
            labels = batch_data['labels']['M'].to(args.device)
            control = batch_data.get('control', None)
            if control is not None:
                control = control.to(args.device)

            labels = labels.view(-1).long() if args.train_mode == 'classification' else labels.view(-1, 1)
            outputs = model(text, audio, vision, control)['M']
            loss = criterion(outputs, labels)

            eval_loss += loss.item()
            y_pred.append(outputs.cpu())
            y_true.append(labels.cpu())

        avg_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = metrics(pred, true)
        eval_results["Loss"] = round(avg_loss, 4)

        print(f"Evaluation Results for {mode} mode: {eval_results}")
        print(f"Predicted: {pred[0].item()}, True: {true[0].item()}")
