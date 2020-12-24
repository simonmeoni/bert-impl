import click

from src.bert_impl.train.fine_tuning_bert import fine_tune_bert_model
from src.bert_impl.train.pretraining_bert import pretrain_bert_model


@click.command()
@click.option('--train-path')
@click.option('--ft-train-path')
@click.option('--pretrain-model-path')
@click.option('--save-model-path')
@click.option('--sp-path')
@click.option('--neptune-api-token')
@click.option('--ft-lr')
@click.option('--batch-size')
@click.option('--epochs')
@click.option('--folds')
@click.option('--corpus-size')
# pylint: disable=too-many-arguments
def fine_tune_bert(train_path,
                   ft_train_path,
                   pretrain_model_path,
                   save_model_path,
                   sp_path,
                   neptune_api_token,
                   ft_lr,
                   batch_size,
                   epochs,
                   folds,
                   corpus_size
                   ):
    click.echo('Training started...')
    # pylint: disable=duplicate-code
    fine_tune_bert_model(train_path,
                         ft_train_path,
                         pretrain_model_path,
                         save_model_path,
                         sp_path,
                         neptune_api_token,
                         ft_lr,
                         batch_size,
                         epochs,
                         folds,
                         corpus_size)
    click.echo('Training finished...')
    click.echo('models are saved to' + save_model_path)


@click.command()
@click.option('--train-path')
@click.option('--test-path')
@click.option('--pretrain-path')
@click.option('--checkpoint-path')
@click.option('--sp-path')
@click.option('--neptune-api-token')
@click.option('--stack-size')
@click.option('--bert-dim-model')
@click.option('--head-size')
@click.option('--pt-lr')
@click.option('--batch-size')
@click.option('--epochs')
@click.option('--corpus-size')
# pylint: disable=too-many-arguments
def pretrain_bert(train_path,
                  test_path,
                  pretrain_path,
                  checkpoint_path,
                  sp_path,
                  neptune_api_token,
                  stack_size,
                  bert_dim_model,
                  head_size,
                  pt_lr,
                  batch_size,
                  epochs,
                  corpus_size
                  ):
    click.echo('pretraining started...')
    # pylint: disable=duplicate-code
    pretrain_bert_model(train_path,
                        test_path,
                        pretrain_path,
                        checkpoint_path,
                        sp_path,
                        neptune_api_token,
                        stack_size,
                        bert_dim_model,
                        head_size,
                        pt_lr,
                        batch_size,
                        epochs,
                        corpus_size)
    click.echo('pretraining finished...')
    click.echo('models are saved to' + checkpoint_path)


@click.command()
def train_albert():
    click.echo('not yet implemented')


@click.command()
def eval_model():
    click.echo('not yet implemented !')


@click.group()
def cli():
    pass


cli.add_command(fine_tune_bert)
cli.add_command(pretrain_bert)
cli.add_command(train_albert)
cli.add_command(eval_model)

if __name__ == '__main__':
    cli()
