import argparse
from helpers import train, test

def main():
    """
    Main function for running cartpole

        Parameters:
            none

        Returns:
            none
    """
    parser = argparse.ArgumentParser(description='Cartpole simulation actions')
    parser.add_argument("-train", "--train",
                        help="visualize and train",
                        action="store_true")
    parser.add_argument("-test", "--test",
                        help="visualize and test",
                        action="store_true")
    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()

if __name__=="__main__":
    main()
