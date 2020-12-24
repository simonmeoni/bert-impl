import numpy as np
import torch
from torch import nn

from src.bert_impl.utils.utils import generate_batches, generate_batched_masked_lm, \
    decode_sel_vector, get_checkpoint_filename, remove_checkpoints


class PreTrainingClassifier(nn.Module):
    def __init__(self, zn_size, output_size):
        super().__init__()
        self.l_1 = nn.Linear(zn_size, output_size)

    def forward(self, z_n):
        return self.l_1(z_n)


def pre_train_loop(neptune, dataset, checkpoint_path, **parameters):
    device = parameters['device']
    losses = []
    masked_classifier = PreTrainingClassifier(parameters['bert_dim_model'],
                                              parameters['vocabulary_size']).to(device)
    next_sentence_classifier = PreTrainingClassifier(parameters['bert_dim_model'], 2).to(device)
    for _ in range(parameters['epochs']):
        # train loop
        for batch in generate_batches(dataset,
                                      parameters['batch_size'],
                                      device=device):
            # Step 4: Propagate the loss signal backward
            y_pred, y_target, masked_loss = masked_task(batch, dataset, masked_classifier,
                                                        **parameters)
            res_loss = masked_loss + sentence_prediction_task(batch, dataset,
                                                              next_sentence_classifier,
                                                              **parameters)
            res_loss.backward()
            # Step 5: Trigger the optimizer to perform one update
            parameters['optimizer'].step()
            neptune.log_metric('pt loss', res_loss.item())
            raw_text_observed = dataset.sentence_piece \
                .Decode([id_obv for id_obv in torch.argmax(y_pred, dim=2)[-1].tolist()
                         if id_obv != dataset.get_mask()])
            neptune.send_text('raw pre-train text observed', raw_text_observed)
            neptune.send_text('pt text expected',
                              dataset.sentence_piece.Decode(y_target[-1].tolist()))
            losses.append(res_loss.item())
        remove_checkpoints(checkpoint_path)
        torch.save(parameters,
                   get_checkpoint_filename(id_fold=1, path=checkpoint_path))
    return np.mean(losses)


def masked_task(batch, dataset, masked_classifier, **parameters):
    model = parameters['model']
    loss = parameters['loss']
    device = parameters['device']
    words_emb_masked = generate_batched_masked_lm(batch['words_embedding'],
                                                  dataset).to(device)
    sentence_emb = batch['sentence_embedding']
    y_target = batch['words_embedding'].to(device)
    # Step 1: Clear the gradients
    model.zero_grad()
    # Step 2: Compute the forward pass of the model
    y_pred = masked_classifier(model(words_emb_masked, sentence_emb))
    # Step 3: Compute the loss value that we wish to optimize
    res_loss = loss(y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
    return y_pred, y_target, res_loss


def replace_st(words_emb, sentiment, dataset):
    replace_word_emb = words_emb.clone().detach()
    replace_word_emb[1] = dataset.get_st_vocab(sentiment)
    return replace_word_emb


def sentence_prediction_task(batch, dataset, next_sentence_classifier, **parameters):
    model = parameters['model']
    loss = parameters['loss']
    device = parameters['device']
    sentence_emb = batch['sentence_embedding']
    words_emb = batch['words_embedding']
    generate_st = [dataset.get_sentiment_i(np.random.choice(dataset.st_voc))
                   for _ in range(words_emb.size(0))]
    y_target = torch.LongTensor([1 if st[0] == st[1] else 0
                                 for st in zip(batch['sentiment_i'], generate_st)]).to(device)
    replaced_word_emb = torch.stack([seq[0] if seq[1] == seq[2] else
                                     replace_st(seq[0], seq[2], dataset) for seq
                                     in zip(words_emb, y_target, generate_st)]).to(device)
    y_pred = next_sentence_classifier(model(replaced_word_emb, sentence_emb)[:, 0].to(device))
    res_loss = loss(y_pred, y_target)
    return res_loss


class FineTuningClassifier(nn.Module):
    def __init__(self, zn_size):
        super().__init__()
        self.conv1d = nn.Conv1d(zn_size, 2, kernel_size=3, padding=1)

    def forward(self, z_n):
        return self.conv1d(z_n)


def fine_tuning_loop(neptune, dataset, train=True, **parameters):
    device = parameters['device']
    model = parameters['model']
    losses = []
    fine_tuning_classifier = FineTuningClassifier(parameters['bert_dim_model']).to(device)
    for _ in range(parameters['epochs'] if train else 1):
        # train loop
        for batch in generate_batches(dataset,
                                      parameters['batch_size'],
                                      device=device):
            y_target = batch['selected_vector'].to(device)
            # Step 1: Clear the gradients
            model.zero_grad()
            # Step 2: Compute the forward pass of the model
            y_pred = fine_tuning_classifier(
                model(batch['words_embedding'], batch['sentence_embedding']).transpose(2, 1))
            # Step 3: Compute the loss value that we wish to optimize
            res_loss = parameters['loss'](y_pred.transpose(1, 2).view(-1, 2), y_target.reshape(-1))
            if train:
                # Step 4: Propagate the loss signal backward
                res_loss.backward()
                # Step 5: Trigger the optimizer to perform one update
                parameters['st_optimizer'].step()
            observed_text = decode_sel_vector(batch['words_embedding'][-1],
                                              dataset, y_pred.transpose(1, 2)[-1].argmax(dim=-1))
            expected_text = decode_sel_vector(batch['words_embedding'][-1], dataset, y_target[-1])
            neptune.log_metric('st loss' if train else 'st eval loss', res_loss.item())
            neptune.send_text('selected text expected'
                              if train else 'selected eval text expected', expected_text)
            neptune.send_text('selected text observed'
                              if train else 'selected eval text observed', observed_text)

            losses.append(res_loss.item())
    return np.mean(losses)
