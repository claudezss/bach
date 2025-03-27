import typer
from huggingface_hub import upload_file

app = typer.Typer()


@app.command()
def upload_transformer_model(file_path: str):
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="music_transformer.pt",
        repo_id="claudezss/bach",
        commit_message="upload transformer model artifact",
        repo_type="model",
    )


@app.command()
def upload_rnn_model(file_path: str):
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="music_rnn.pt",
        repo_id="claudezss/bach",
        commit_message="upload rnn model artifact",
        repo_type="model",
    )


if __name__ == "__main__":
    app()
