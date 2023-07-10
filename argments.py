import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RCE')
    parser.add_argument('--env-name', type=str,default='door-human-v0')
    parser.add_argument('--device', type=str,default='cpu')
    parser.add_argument('--n-steps', type=int, default=10)
    parser.add_argument('--critic-loss', type=str, default='c')
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--log-std-min', type=float, default=-2.)
    parser.add_argument('--log-std-max', type=float, default=-20.)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=1e-4)
    parser.add_argument('--rollout-steps', type=int, default=201)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--evaluation_rollouts', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--save-model-interval', type=int, default=10000)
    parser.add_argument('--display-loss-interval', type=int, default=1000)
    parser.add_argument('--init_num_n_step_trajs', type=int, default=1000)
    parser.add_argument('--polyak', type=float, default=0.95)

    args = parser.parse_args()
    return args


