import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--simple_model', type=str, default='none', help='simple-rec-model, support [none, lgn-ide, gf-cf, bspm]')

    parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--solver_idl', type=str, default='euler', help="heat equation solver")
    parser.add_argument('--solver_blr', type=str, default='euler', help="ideal low-pass solver")
    parser.add_argument('--solver_shr', type=str, default='euler', help="sharpening solver")

    parser.add_argument('--K_idl', type=int, default=1, help='T_idl / \tau')
    parser.add_argument('--T_idl', type=float, default=1, help='T_idl')

    parser.add_argument('--K_b', type=int, default=1, help='T_b / \tau')
    parser.add_argument('--T_b', type=float, default=1, help='T_b')

    parser.add_argument('--K_s', type=int, default=1, help='T_s / \tau')
    parser.add_argument('--T_s', type=float, default=1, help='T_s')

    parser.add_argument('--factor_dim', type=int, default=256, help='factor_dim')
    parser.add_argument('--idl_beta', type=float, default=0.3, help='beta')

    parser.add_argument('--final_sharpening', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sharpening_off', type=eval, default=False, choices=[True, False])
    parser.add_argument('--t_point_combination', type=eval, default=False, choices=[True, False])

    parser.add_argument('--gpu', type=int, default=0, help='0')


    return parser.parse_args()
