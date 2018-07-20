import joblib
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')

    args = parser.parse_args()


    ### Part uno
    # pkl = args.file

    # data = joblib.load(pkl)
    # policy = data['policy']
    # func = policy._f_dist

    # file_name = "function.pkl"
    # joblib.dump(func, file_name, compress=3)


    ### Part dos

    pkl = args.file

    data = joblib.load(pkl)
    import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    main()