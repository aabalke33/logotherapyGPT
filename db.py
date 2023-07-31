from chromadb.config import Settings


class db:
    def __init__(self, name, root):
        self.name = name
        self.directory = f"{root}/DB/DB_{name}"
        self.chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.directory,
            anonymized_telemetry=False
        )

    def get_name(self):
        return self.name

    def get_directory(self):
        return self.directory

    def get_chroma_settings(self):
        return self.chroma_settings
