import torch
import argparse
import json
from Config import config
from Train import Trainer
from Inference import Inferencer
from DataProcessor import DataProcessor

def train(train_chrom_list, valid_chrom_list, valid_ratio):
    # default value
    window_size = 2048
    overlap = 1024
    batch_size = 16
    lr = 1e-4
    num_workers = 8
    epochs = 500
    weight_decay = 1e-2
    d_model = 128
    n_head = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    k = 6
    local_window = 256
    save_path = 'Model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load configuration
    model_config_file_path = "config/modelconfig.json"
    train_config_file_path = "config/trainconfig.json"
    c = config()
    try:
        modelconfig = c.load_configuration(model_config_file_path)
        trainconfig = c.load_configuration(train_config_file_path)
        print("Configuration loaded successfully.")
        # model config setup
        window_size = modelconfig["window_size"]
        overlap = modelconfig["overlap"]
        batch_size = modelconfig["batch_size"]
        d_model = modelconfig["token_dimension"]
        n_head = modelconfig["head_count"]
        num_layers = modelconfig["transformer_layer_count"]
        dim_feedforward = modelconfig["feedforward_dimension"]
        dropout = modelconfig["dropout"]
        k = modelconfig["mers_count"]
        local_window = modelconfig["local_attention_window"]
        save_path = modelconfig["model_save_path"]
        # train config setup
        lr = trainconfig["learning_rate"]
        num_workers = trainconfig["num_workers"]
        epochs = trainconfig["epochs"]
        weight_decay = trainconfig["weight_decay"]
        ref_gene_file = trainconfig["ref_gene_file"]
        gene_signal_gile = trainconfig["gene_signal_file"]
    except Exception:
        print("Cannot load configuration, use default parameters.")
    # process data
    dp = DataProcessor()
    dp.add_data_from_bigwig(ref_gene_file, gene_signal_gile)
    # train_dataset, valid_dataset = dp.generate_dataset(window_size, overlap, train_chrom_list, valid_chrom_list, 0.5)
    train_dataset, valid_dataset = dp.generate_one_chrom_dataset(window_size, overlap, train_chrom_list[0], valid_ratio)
    # make logger
    logger = ["ws"+str(window_size), "ol"+str(overlap), "df"+str(dim_feedforward), "la"+str(local_window)]
    log_name = "_".join(_ for _ in logger)+".txt"
    # train
    tr = Trainer(train_dataset=train_dataset,
                 valid_dataset=valid_dataset,
                 window_size=window_size,
                 overlap=overlap,
                 batch_size=batch_size,
                 lr=lr,
                 num_workers=num_workers,
                 epochs=epochs,
                 weight_decay=weight_decay,
                 d_model=d_model,
                 n_head=n_head,
                 num_layers=num_layers,
                 dim_feedforward=dim_feedforward,
                 dropout=dropout,
                 k=k,
                 local_window=local_window,
                 save_path=save_path,
                 device=device,
                 logger=log_name)
    tr.train()

def inference(infer_chrom, infer_start, infer_end):
    # default value
    window_size = 2048
    overlap = 1024
    d_model = 128
    n_head = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    k = 6
    save_path = 'Model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_file_path = "config/modelconfig.json"
    infer_config_file_path = "config/inferconfig.json"
    c = config()
    try:
        modelconfig = c.load_configuration(model_config_file_path)
        inferconfig = c.load_configuration(infer_config_file_path)
        print("Configuration loaded successfully.")
        # model config setup
        window_size = modelconfig["window_size"]
        overlap = modelconfig["overlap"]
        d_model = modelconfig["token_dimension"]
        n_head = modelconfig["head_count"]
        num_layers = modelconfig["transformer_layer_count"]
        dim_feedforward = modelconfig["feedforward_dimension"]
        dropout = modelconfig["dropout"]
        k = modelconfig["mers_count"]
        save_path = modelconfig["model_save_path"]
        # infer config setup
        ref_gene_file = inferconfig["ref_gene_file"]
        gene_signal_gile = inferconfig["gene_signal_file"]

    except Exception:
        print("Cannot load configuration, use default parameters.")

    # process infer data
    dp = DataProcessor()
    dp.add_data_from_bigwig(ref_gene_file, gene_signal_gile)
    token_ids = dp.get_token_ids_by_chrom(infer_chrom, infer_start, infer_end)
    tgt_signals = dp.get_tgt_signal_by_chrom(infer_chrom, infer_start, infer_end)
    # tensonization
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # inference
    ifc = Inferencer(
        token_ids=token_ids,
        window_size=window_size,
        overlap=overlap,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        k=k,
        save_path=save_path,
        device=device,
        unlog1pscale=True
    )
    pred = ifc.inference()
    return pred, tgt_signals

def visualize(pred, tgt):
    # visualize
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(tgt)
    axs[0].set_title('target')
    axs[1].plot(pred)
    axs[1].set_title('predict')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train',
                        help="train mode: train chromosome id list and validation list ratio required.\
                            example usage: --train --train_chroms Chr1 --valid_ratio 0.2",
                        action='store_true')
    parser.add_argument('-tl','--train_chroms',
                        type=str,
                        dest='tl',
                        nargs='+',
                        help="add chroms used in training.\
                            example usage: -tl Chr1 Chr2 Chr3")
    parser.add_argument('-va','--valid_ratio',
                        type=float,
                        dest='va',
                        default=0.2,
                        help="add the ratio of dataset will be used for vak=lidation.\
                            example usage: -va 0.1")
    parser.add_argument('-i','--infer',
                        help="infer mode: infer chromosome id.\
                            example usage: --infer Chr1 0 1000000",
                        action='store_true')
    parser.add_argument('infer_chrom', type=str,help='chrom id for inference')
    parser.add_argument('infer_start', type=int, help='start index of chromosome for interence')
    parser.add_argument('infer_end', type=int, help='end index of chromosome for interence')
    args = parser.parse_args()
    if args.train:
        train(train_chrom_list=args.tl, valid_chrom_list=[], valid_ratio=args.va)
    elif args.infer:
        pred, tgt = inference(args.infer_chrom, args.infer_start, args.infer_end)
        visualize(pred,tgt)



