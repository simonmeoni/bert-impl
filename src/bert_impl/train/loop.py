import torch
import torch.nn.functional as f
from torch import nn

from src.bert_impl.utils.utils import generate_batches, generate_batched_masked_lm


class PreTrainingClassifier(nn.Module):
    def __init__(self, zn_size, voc_size):
        super().__init__()
        self.l_1 = nn.Linear(zn_size, voc_size)

    def forward(self, z_n):
        return self.l_1(z_n)


def pre_train_loop(neptune, dataset, train=True, **parameters):
    device = parameters['device']
    model = parameters['model']

    pre_train_classifier = PreTrainingClassifier(parameters['bert_dim_model'],
                                                 parameters['vocabulary_size']).to(device)
    for _ in range(parameters['epochs']):
        # train loop
        dataset.switch_to_dataset("train")
        for batch in generate_batches(dataset,
                                      parameters['batch_size'],
                                      device=device):

            x_obs = generate_batched_masked_lm(batch['vectorized_tokens'],
                                               dataset).to(device)
            y_target = batch['vectorized_tokens'].to(device)
            # Step 1: Clear the gradients
            model.zero_grad()
            # Step 2: Compute the forward pass of the model
            y_pred = pre_train_classifier(model(x_obs))
            # Step 3: Compute the loss value that we wish to optimize
            res_loss = parameters['loss'](y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
            if train:
                # Step 4: Propagate the loss signal backward
                res_loss.backward()
                # Step 5: Trigger the optimizer to perform one update
                parameters['optimizer'].step()
            neptune.log_metric('pre-train loss', res_loss.item())
            observed_ids = torch.argmax(y_pred, dim=2)[-1]
            raw_text_observed = dataset.sentence_piece \
                .Decode([id_obv for id_obv in observed_ids.tolist()
                         if id_obv != dataset.get_mask()])
            neptune.send_text('raw pre-train text observed'
                              if train else 'raw eval pre-train text observed', raw_text_observed)
            neptune.send_text('raw pre-train text expected' if train else
                              'raw eval pre-train text expected',
                              dataset.sentence_piece.Decode(y_target[-1].tolist()))


class FineTuningClassifier(nn.Module):
    def __init__(self, zn_size, st_voc_size, voc_size):
        super().__init__()
        self.l_1 = nn.Linear(zn_size, voc_size)
        self.l_2 = nn.Linear(voc_size, st_voc_size)

    def forward(self, z_n):
        l1_out = f.relu((self.l_1(z_n)))
        out = self.l_2(l1_out)
        return out


def fine_tuning_loop(neptune, dataset, train=True, **parameters):
    device = parameters['device']
    model = parameters['model']
    loss = parameters['loss']
    optimizer = parameters['optimizer']

    fine_tuning_classifier = FineTuningClassifier(parameters['bert_dim_model'],
                                                  len(dataset.st_voc),
                                                  parameters['vocabulary_size']).to(device)
    for _ in range(parameters['epochs']):
        # train loop
        dataset.switch_to_dataset("train")
        for batch in generate_batches(dataset,
                                      parameters['batch_size'],
                                      device=device):
            x_obs = batch['vectorized_tokens'].to(device)
            y_target = batch['sentiment_i'].to(device)
            # Step 1: Clear the gradients
            model.zero_grad()
            # Step 2: Compute the forward pass of the model
            y_pred = fine_tuning_classifier(model(x_obs)[:, -1, :])
            # Step 3: Compute the loss value that we wish to optimize
            res_loss = loss(y_pred, y_target.reshape(-1))
            if train:
                # Step 4: Propagate the loss signal backward
                res_loss.backward()
                # Step 5: Trigger the optimizer to perform one update
                optimizer.step()
            neptune.log_metric('sentiment train loss', res_loss.item())
            neptune.send_text('sentiment train text',
                              dataset.sentence_piece.Decode(x_obs[-1].tolist()))
            neptune.send_text('sentiment train observed',
                              dataset.st_voc[torch.argmax(y_pred, dim=-1)[-1]])
            neptune.send_text('sentiment train expected',
                              dataset.st_voc[y_target[-1]])
