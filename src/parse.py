import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default='/')
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--task_id", type=str, default="")

parser.add_argument("--dataset", type=str, default='cifar10', help='options: cifar10, cifar100')
parser.add_argument("--data_size", type=str, default='small', help='options: small, full')
parser.add_argument("--data_partition", type=str, default='iid', help='options: class, iid, dir')
parser.add_argument("--dir_alpha", type=float, default=0.5, help='alpha for dirichlet sampling')
parser.add_argument('--client_model', default='byol', type=str, help='options: byol, simsiam, simclr, moco, moco_v2')
# TODO: list server model and clients model
parser.add_argument('--server_model', default='fedet', type=str, help='options: byolserver, fedmd, single')
parser.add_argument('--client_type', default='mix', type=str, help='options: resnet18, resnet50, mix, vgg')
parser.add_argument('--framework', default='fedet', type=str, help='options: ours, fedavg, fedu, oursnoalign, single')
parser.add_argument('--public', default='iid', type=str, help='options: iid,class')


parser.add_argument('--encoder_network', default='resnet18', type=str,
                help='network architecture of server encoder, options: resnet18, resnet50')
parser.add_argument('--predictor_network', default='2_layer', type=str,
                help='network of predictor, options: 1_layer, 2_layer')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--local_epoch', default=1, type=int)  # local epoch
parser.add_argument('--server_epoch', default=1, type=int)  # server epoch
parser.add_argument('--rounds', default=1, type=int)  # global epoch
parser.add_argument('--num_of_clients', default=5, type=int)
parser.add_argument('--clients_per_round', default=5, type=int)
parser.add_argument('--class_per_client', default=2, type=int,
                help='for non-IID setting, number of classes each client, based on CIFAR10')
parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.032, type=float)
parser.add_argument('--lr_type', default='cosine', type=str, help='cosine decay learning rate')
parser.add_argument('--random_selection', action='store_true', help='whether randomly select clients')
parser.add_argument('--aggregate_encoder', default='online', type=str, help='options: online, target')
parser.add_argument('--update_encoder', default='online', type=str, help='options: online, target, both, none')
parser.add_argument('--update_predictor', default='global', type=str, help='options: global, local, dapu')
parser.add_argument('--dapu_threshold', default=0.4, type=float, help='DAPU threshold value')
parser.add_argument('--weight_scaler', default=1.0, type=float, help='weight scaler for different class per client')
parser.add_argument('--auto_scaler', default='y', type=str, help='use value to compute auto scaler')
parser.add_argument('--auto_scaler_target', default=0.8, type=float,
                    help='target weight for the first time scaling')
parser.add_argument('--encoder_weight', type=float, default=0,
                help='for ema encoder update, apply on local encoder')
parser.add_argument('--predictor_weight', type=float, default=0,
                help='for ema predictor update, apply on local predictor')

parser.add_argument('--test_every', default=1, type=int, help='test every x rounds')
parser.add_argument('--save_model_every', default=1, type=int, help='save model every x rounds')
parser.add_argument('--save_model_path', default="", type=str, help='save model every x rounds')
parser.add_argument('--save_predictor', action='store_true', help='whether save predictor')
parser.add_argument('--semi_supervised', action='store_true', help='whether to train with semi-supervised data')
parser.add_argument('--label_ratio', default=0.01, type=float, help='percentage of labeled data')

parser.add_argument('--gpu', default=['2'], nargs='+')
parser.add_argument('--run_count', default=2, type=int)
parser.add_argument('--public_size', default=4000, type=int)
parser.add_argument('--client_num', default=0, type=int)

args = parser.parse_args()
print("arguments: ", args)
