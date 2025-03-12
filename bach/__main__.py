import typer

from bach.data import app as data_app

app = typer.Typer()
app.add_typer(data_app, name="data")

if __name__ == "__main__":
    app()
