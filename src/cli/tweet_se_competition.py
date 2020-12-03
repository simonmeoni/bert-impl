import click
from src.bert_impl.train.tweet_extraction_bert import train_bert_model


@click.command()
@click.option('--train-path')
@click.option('--pr-train-path')
@click.option('--checkpoint-path')
@click.option('--sp-path')
@click.option('--neptune-api-token')
@click.option('--stack-size')
@click.option('--bert-dim-model')
@click.option('--head-size')
@click.option('--pt-lr')
@click.option('--st-lr')
@click.option('--batch-size')
@click.option('--epochs')
@click.option('--folds')
@click.option('--corpus-size')
# pylint: disable=too-many-arguments
def train_bert(train_path,
               pr_train_path,
               checkpoint_path,
               sp_path,
               neptune_api_token,
               stack_size,
               bert_dim_model,
               head_size,
               pt_lr,
               st_lr,
               batch_size,
               epochs,
               folds,
               corpus_size
               ):
    click.echo('Training started...')
    # pylint: disable=duplicate-code
    train_bert_model(train_path,
                     pr_train_path,
                     checkpoint_path,
                     sp_path,
                     neptune_api_token,
                     stack_size,
                     bert_dim_model,
                     head_size,
                     pt_lr,
                     st_lr,
                     batch_size,
                     epochs,
                     folds,
                     corpus_size)
    click.echo('Training finished...')
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


cli.add_command(train_bert)
cli.add_command(train_albert)
cli.add_command(eval_model)

if __name__ == '__main__':
    cli()
