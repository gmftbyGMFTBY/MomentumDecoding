import json
import numpy as np
import args

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = json.load(open(args.test_path))
    times = []
    for session in data:
        times.extend(session['time_cost'])
    print(f'[!] average time cost: {np.mean(times)}')
