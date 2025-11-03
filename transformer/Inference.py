import torch
import torch.nn.functional as F
from GenomicTransformModel import GenomicTransformModel

class Inferencer():
    def __init__(
            self,
            token_ids=None,
            window_size=1024,
            overlap=512,
            d_model=128,
            n_head=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            k=6,
            save_path='Model.pth',
            device=torch.device('cuda'),
            unlog1pscale=False
    ):
        self.token_ids = token_ids
        self.window_size = window_size
        self.overlap = overlap
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.k = k
        self.save_path = save_path
        self.stride = window_size - overlap
        self.device = device
        self.unlog1pscale = unlog1pscale

    def inference(self):
        # need another model instance to inference
        model = GenomicTransformModel(vocab_size=4 ** self.k, 
                                      d_model=self.d_model,
                                      nhead=self.n_head,
                                      num_layers=self.num_layers,
                                      dim_feedforward=self.dim_feedforward,
                                      dropout=self.dropout,
                                      window_size=self.window_size,
                                      gradient_checkpoint=True)
        model.to(self.device)
        checkpoint = torch.load(self.save_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        full_length = len(self.token_ids)
        pred_sum = torch.zeros(full_length, device=self.device)
        count = torch.zeros(full_length, device=self.device)

        with torch.no_grad():
            for start in range(0, full_length, self.stride):
                end = min(start + self.window_size, full_length)
                seq_window = self.token_ids[start:end].to(self.device)
                if end - start < self.window_size:
                    pad_len = self.window_size - (end - start)
                    seq_window = F.pad(seq_window, (0, pad_len), value=0)
                seq_window = seq_window.unsqueeze(0)
                window_pred = model(seq_window)
                window_pred = window_pred.squeeze(0)
                # unscale log1p
                if self.unlog1pscale:
                    window_pred = torch.expm1(window_pred)
                # if padded, trim the prediction
                window_pred = window_pred[:(end-start)]
                # accumulate results
                pred_sum[start:end] += window_pred
                count[start:end] += 1

        full_sequence_pred = pred_sum / count
        full_sequence_pred = full_sequence_pred.cpu().numpy()

        return full_sequence_pred

if __name__ == '__main__':
    ifc = Inferencer()
    ifc.inference()
