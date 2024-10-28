import argparse

def parameter_parser():
    
    parser = argparse.ArgumentParser(description="Run LightSGCN.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/bitcoin_otc.csv",
	                help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/bitcoin_otc.csv",
	                help="Edge list csv.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")
    
    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
	                help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
	                help="Number of SVD feature extraction dimensions. Default is 64.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
	                help="Test dataset size. Default is 0.2.")
    
    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")

    parser.add_argument("--num_layers",
                        type=int,
                        default=32,
                    help="Number of layers. Default is 32.")
    
    parser.add_argument("--alpha_0",
                        type=float,
                        default=0.5,
	                help="alpha_0. Default is 0.5.")

    parser.add_argument("--alpha_k",
                        type=float,
                        default=0.5,
	                help="alpha_k. Default is 0.5.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
	                help="Learning rate. Default is 10^-5.")

    parser.add_argument("--out_features",
                        type=int,
                        default=32,
	                help="out_features. Default is 32.")
    
    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 100.")

    parser.add_argument("--beta",
                        type=float,
                        default=0.7,
	                help="beta. Default is 0.7.")
    
    parser.add_argument("--lamb",
                        type=float,
                        default=0.0,
	                help="lamb. Default is 0.0.")

    parser.add_argument("--log-path",
                        nargs="?",
                        default="./logs/bitcoin_otc_logs.json",
	                help="Log json.")

    parser.set_defaults(spectral_features=True)

    return parser.parse_args()
