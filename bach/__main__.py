import typer

from bach.data import app as data_app
from bach.inference import app as inference_app
from bach.train.train_rnn import app as train_rnn_app
from bach.train.train_transformer import app as train_transformer_app
from bach.upload import app as upload_app

app = typer.Typer()
app.add_typer(data_app, name="data")
app.add_typer(upload_app, name="artifact")
app.add_typer(train_rnn_app, name="rnn")
app.add_typer(train_transformer_app, name="transformer")
app.add_typer(inference_app, name="infer")

if __name__ == "__main__":
    app()
