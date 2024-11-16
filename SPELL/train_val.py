import os
import torch
import numpy as np
import random
import argparse
import pandas as pd
from data_loader import AVADataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score


parser = argparse.ArgumentParser(description='SPELL')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run the train_val')
parser.add_argument('--feature', type=str, default='resnet18-tsm-aug', help='name of the features')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes (n in our paper)')
parser.add_argument('--time_edge', type=float, default=0.9, help='time threshold (tau in our paper)')
parser.add_argument('--cross_identity', type=str, default='cin', help='whether to allow cross-identity edges')
parser.add_argument('--edge_weight', type=str, default='fsimy', help='how to decide edge weights')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # 5e-4 or 1e-3 works well
parser.add_argument('--sch_param', type=int, default=100, help='parameter for lr scheduler') # 10 or 100
parser.add_argument('--channel1', type=int, default=64, help='filter dimension of GCN layers (layer1-2)')
parser.add_argument('--channel2', type=int, default=16, help='filter dimension of GCN layers (layer2-3)')
parser.add_argument('--proj_dim', type=int, default=64, help='projection of 4->proj_dim for spatial feature')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for SAGEConv') # 0.2 ~ 0.4
parser.add_argument('--dropout_a', type=float, default=0, help='dropout value for dropout_adj')
parser.add_argument('--da_true', action='store_true', help='always apply dropout_adj for both the training and testing')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--num_epoch', type=int, default=60, help='total number of epochs')
parser.add_argument('--eval_freq', type=int, default=1, help='how frequently run the evaluation')
parser.add_argument('--type', type=str, default='test', help="train or test or val")
parser.add_argument('--save_predict_csv_dir', type=str, default='test', help="Model checkpoint path")
parser.add_argument('--GNNType', type=str, default='OriginalGNN', help="Model checkpoint path")
parser.add_argument('--model_pt_file', type=str, default='chckpoint_best_map.pt', help="Model checkpoint path")
parser.add_argument('--student_data_path', type=str, default='../student_data/student_data', help="Model checkpoint path")



args = parser.parse_args()

if args.GNNType == 'OriginalGNN':
    from models_gnn_original import SPELL
    args.model_pt_file = 'chckpoint_best_map.pt'

elif args.GNNType == 'TripleLyearsGNN':
    from models_gnn_doubleLayer import SPELL
    args.model_pt_file = 'TripleLyearsGNN.pt'

elif args.GNNType == 'SingleMSU':
    from models_gnn_singleMSU import SPELL
    args.model_pt_file = 'SingleUnet.pt'

elif args.GNNType == 'SingleUnet':
    from models_gnn_singleUnet import SPELL
    args.model_pt_file = 'SingleUnet.pt'

elif args.GNNType == 'TripleMSU':
    from models_gnn_TripleMSU import SPELL
    args.model_pt_file = 'TripleMSU.pt'
print("Now using GNN type of : ", args.GNNType)


def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph_data = {}
    graph_data['numv'] = args.numv
    graph_data['skip'] = graph_data['numv']
    graph_data['time_edge'] = args.time_edge
    graph_data['cross_identity'] = args.cross_identity
    graph_data['edge_weight'] = args.edge_weight

    # path of the audio-visual features
    dpath_root = os.path.join('features', '{}'.format(args.feature))

    # path of the generated graphs
    exp_key = '{}_{}_{}_{}_{}'.format(args.feature, graph_data['numv'], graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight'])
    tpath_root = os.path.join('graphs', exp_key)

    # path for the results and model checkpoints
    exp_name = '{}_lr{}-{}_c{}-{}_d{}-{}_s{}'.format(exp_key, args.lr, args.sch_param, args.channel1, args.channel2, args.dropout, args.dropout_a, args.seed)

    print (exp_name)

    result_path = os.path.join('models', exp_name)
    os.makedirs(result_path, exist_ok=True)

    dpath_train = os.path.join(dpath_root, 'train_forward', '*.csv')
    tpath_train = os.path.join(tpath_root, 'train')
    dpath_val = os.path.join(dpath_root, 'val_forward', '*.csv')
    tpath_val = os.path.join(tpath_root, 'val')

    cont = 1
    Fdataset_train = AVADataset(dpath_train, graph_data, cont, tpath_train, mode='train')
    Fdataset_val = AVADataset(dpath_val, graph_data, cont, tpath_val, mode='val')

    train_loader = DataLoader(Fdataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Fdataset_val, batch_size=1, shuffle=False, num_workers=4)

    # gpu and learning parameter settings
    feature_dim = 1024
    if 'resnet50' in args.feature:
        feature_dim = 4096

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    model = SPELL([args.channel1, args.channel2], feature_dim, args.dropout, args.dropout_a, args.da_true, proj_dim=args.proj_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sch_param)

    flog = open(os.path.join(result_path, 'log.txt'), mode = 'w')
    max_mAP = 0
    max_acc = 0
    for epoch in range(1, args.num_epoch+1):
        loss = train(model, train_loader, device, optimizer, criterion, scheduler)
        str_print = '[{:3d}|{:3d}]: Training loss: {:.4f}'.format(epoch, args.num_epoch, loss)

        if epoch % args.eval_freq == 0:
            mAP, acc = evaluation(model, val_loader, device, feature_dim)
            if mAP > max_mAP:
                max_mAP = mAP
                epoch_max = epoch
                # torch.save(model.state_dict(), os.path.join(result_path, 'chckpoint_{:03d}.pt'.format(epoch)))
                torch.save(model.state_dict(), os.path.join(result_path, args.model_pt_file))
            # if acc > max_acc:
            #     max_acc = acc
            #     epoch_max = epoch
            #     torch.save(model.state_dict(), os.path.join(result_path, 'chckpoint_{:03d}.pt'.format(epoch)))

            # str_print += ', mAP: {:.4f} (max_mAP: {:.4f} at epoch: {})'.format(mAP, max_mAP, epoch_max)
            str_print += ', acc: {:.4f} (max_acc: {:.4f} at epoch: {})'.format(acc, max_mAP, epoch_max)

        print (str_print)
        flog.write(str_print+'\n')
        flog.flush()

    flog.close()


def train(model, train_loader, device, optimizer, criterion, scheduler):
    model.train()
    loss_sum = 0.

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, data.y)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    scheduler.step()

    return loss_sum/len(train_loader)


def evaluation(model, val_loader, device, feature_dim):
    model.eval()
    target_total = []
    soft_total = []
    pred_total = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            x = data.x
            y = data.y

            scores = model(data)
            scores = scores[:, 0].tolist()
            preds = [1.0 if i >= 0.5 else 0.0 for i in scores]

            soft_total.extend(scores)
            target_total.extend(y[:, 0].tolist())
            pred_total.extend(preds)

    # it does not produce an official mAP score (but the difference is negligible)
    # we report the scores computed by an official evaluation script by ActivityNet in our paper
    mAP = average_precision_score(target_total, soft_total)
    acc = accuracy_score(target_total, pred_total)

    return mAP, acc


def validate():
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph_data = {}
    graph_data['numv'] = args.numv
    graph_data['skip'] = graph_data['numv']
    graph_data['time_edge'] = args.time_edge
    graph_data['cross_identity'] = args.cross_identity
    graph_data['edge_weight'] = args.edge_weight

    # path of the audio-visual features
    dpath_root = os.path.join('features', '{}'.format(args.feature))

    # path of the generated graphs
    exp_key = '{}_{}_{}_{}_{}'.format(args.feature, graph_data['numv'], graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight'])
    tpath_root = os.path.join('graphs', exp_key)

    # path for the results and model checkpoints
    exp_name = '{}_lr{}-{}_c{}-{}_d{}-{}_s{}'.format(exp_key, args.lr, args.sch_param, args.channel1, args.channel2, args.dropout, args.dropout_a, args.seed)

    print(exp_name)

    result_path = os.path.join('models', exp_name)

    dpath_val = os.path.join(dpath_root, 'val_forward', '*.csv')
    tpath_val = os.path.join(tpath_root, 'val')
    cont = 1
    Fdataset_val = AVADataset(dpath_val, graph_data, cont, tpath_val, mode='val')

    val_loader = DataLoader(Fdataset_val, batch_size=1, shuffle=False, num_workers=4)

    # gpu and learning parameter settings
    feature_dim = 1024
    if 'resnet50' in args.feature:
        feature_dim = 4096

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    model = SPELL([args.channel1, args.channel2], feature_dim, args.dropout, args.dropout_a, args.da_true, proj_dim=args.proj_dim)
    model.to(device)

    ckpt = torch.load(os.path.join(result_path, args.model_pt_file), map_location = device)
    model.load_state_dict(ckpt)

    model.eval()
    target_total = []
    soft_total = []
    pred_total = []

    frame_id_total = []
    identity_total = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            x = data.x
            y = data.y
            # print(data, "here")

            frame_id = data.times     # form: [np_array]
            identity = data.identity

            scores = model(data)
            scores = scores[:, 0].tolist()
            preds = [1 if i >= 0.5 else 0 for i in scores]

            soft_total.extend(scores)
            target_total.extend(y[:, 0].tolist())
            pred_total.extend(preds)
            # here have to ask
            # print(frame_id)
            frame_id_total.extend(frame_id[0].astype(int).tolist())
            identity_total.extend(identity[0].tolist())

    mAP = average_precision_score(target_total, soft_total)
    acc = accuracy_score(target_total, pred_total)
    print(f'mAP: {mAP}, Classification Accuracy: {acc}')

    # Convert to the Kaggle's format
    df_pred = pd.DataFrame({'identity': identity_total, 'frame_id': frame_id_total, 'pred': pred_total})
    
    df_dis = pd.read_csv("../dis.csv")
    val_video_ids = df_dis['video_id'][-20:].values

    output_ids = []
    output_preds = []

    correct, total = 0, 0
    label_sum = 0

    for video_id in val_video_ids:
        seg_file = f"{args.student_data_path}/train/seg/{video_id}_seg.csv"
        df = pd.read_csv(seg_file)     # columns: [person_id, start_frame, end_frame]
        for index, row in df.iterrows():
            if row[0] == "person_id": continue
            pred = df_pred[(df_pred['identity'] == f"{video_id}_{row[0]}") & (df_pred['frame_id'] >= row[1]) & (df_pred['frame_id'] <= row[2])]['pred'].values
            pred = 1 if np.mean(pred) >= 0.5 else 0
            output_ids.append(f"{video_id}_{row[0]}_{row[1]}_{row[2]}")
            output_preds.append(pred)
            # Check validation accuracy
            total += 1
            correct += (pred == row[3])
            label_sum += row[3]

    df_output = pd.DataFrame({
        'Id': output_ids,
        'Predicted': output_preds
    })
    print(df_output)
    print(f"{sum(df_output['Predicted'].values)} / {total} = {sum(df_output['Predicted'].values) / total}")
    print(f"{label_sum} / {total} = {label_sum / total}")
    print(f'Classification Accuracy after Conversion: {correct / total}')


def test():
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph_data = {}
    graph_data['numv'] = args.numv
    graph_data['skip'] = graph_data['numv']
    graph_data['time_edge'] = args.time_edge
    graph_data['cross_identity'] = args.cross_identity
    graph_data['edge_weight'] = args.edge_weight

    # path of the audio-visual features
    dpath_root = os.path.join('features', '{}'.format(args.feature))

    # path of the generated graphs
    exp_key = '{}_{}_{}_{}_{}'.format(args.feature, graph_data['numv'], graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight'])
    tpath_root = os.path.join('graphs', exp_key)

    # path for the results and model checkpoints
    exp_name = '{}_lr{}-{}_c{}-{}_d{}-{}_s{}'.format(exp_key, args.lr, args.sch_param, args.channel1, args.channel2, args.dropout, args.dropout_a, args.seed)

    print(exp_name)

    result_path = os.path.join('models', exp_name)
    print("result_path", result_path)
    dpath_test = os.path.join(dpath_root, 'test_forward', '*.csv')
    tpath_test = os.path.join(tpath_root, 'test')

    cont = 1
    Fdataset_test = AVADataset(dpath_test, graph_data, cont, tpath_test, mode='test')

    test_loader = DataLoader(Fdataset_test, batch_size=1, shuffle=False, num_workers=4)

    # gpu and learning parameter settings
    feature_dim = 1024
    if 'resnet50' in args.feature:
        feature_dim = 4096

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    model = SPELL([args.channel1, args.channel2], feature_dim, args.dropout, args.dropout_a, args.da_true, proj_dim=args.proj_dim)
    model.to(device)

    ckpt = torch.load(os.path.join(result_path, args.model_pt_file), map_location = device)
    model.load_state_dict(ckpt)

    model.eval()
    pred_total = []

    frame_id_total = []
    identity_total = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            frame_id = data.times     # form: [np_array]
            identity = data.identity

            scores = model(data)
            scores = scores[:, 0].tolist()
            preds = [1 if i >= 0.5 else 0 for i in scores]

            pred_total.extend(preds)

            frame_id_total.extend(frame_id[0].astype(int).tolist())
            identity_total.extend(identity[0].tolist())

    # Convert to the Kaggle's format
    df_pred = pd.DataFrame({'identity': identity_total, 'frame_id': frame_id_total, 'pred': pred_total})
    print()
    test_video_ids = sorted(os.listdir(os.path.join(args.student_data_path, args.type, "seg")))
    test_video_ids = [f[:-8] for f in test_video_ids]

    output_ids = []
    output_preds = []

    total = 0

    for video_id in test_video_ids:
        seg_file = f"{args.student_data_path}/{args.type}/seg/{video_id}_seg.csv"
        df = pd.read_csv(seg_file)     # columns: [person_id, start_frame, end_frame]
        for index, row in df.iterrows():
            if row[0] == "person_id": continue
            pred = df_pred[(df_pred['identity'] == f"{video_id}_{row[0]}") & (df_pred['frame_id'] >= row[1]) & (df_pred['frame_id'] <= row[2])]['pred'].values
            pred = 1 if np.mean(pred) >= 0.5 else 0
            output_ids.append(f"{video_id}_{row[0]}_{row[1]}_{row[2]}")
            output_preds.append(pred)

            total += 1

    df_output = pd.DataFrame({
        'Id': output_ids,
        'Predicted': output_preds
    })
    print(f"{sum(df_output['Predicted'].values)} / {total} = {sum(df_output['Predicted'].values) / total}")
    tempdir = os.path.split(args.save_predict_csv_dir)

    os.makedirs(os.path.join(tempdir[0]), exist_ok=True)
    df_output.to_csv(args.save_predict_csv_dir, index = False)


if __name__ == '__main__':
    if args.type == 'train':
        main()
    elif args.type == 'val':
        validate()
    else:
        test()
