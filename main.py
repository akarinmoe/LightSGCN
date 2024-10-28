from param_parser import parameter_parser
from utils import tab_printer, read_graph, read_graph, score_printer, save_logs
from preprocess import PreProcess
from LightSGCN import LightSGCNTrainer

def main():
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    pre = PreProcess(edges, args)
    X, prob_vi, positive_edges, negative_edges, test_positive_edges, test_negative_edges, y = pre.preprocess()
    trainer = LightSGCNTrainer(args, X, prob_vi, positive_edges, negative_edges, test_positive_edges, test_negative_edges, y, edges)
    trainer.create_and_train_model()
    if args.test_size > 0:
        # trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
