from pathlib import Path
from pipelines.finebooru import FinebooruReader, FinebooruMetadataFilter, FinebooruWriter
from base import LocalPipeline
from components.images.base import ImageTransform


if __name__ == "__main__":
    wkdir = Path(".")

    LocalPipeline(
        reader=FinebooruReader(wkdir),
        steps=[
            FinebooruMetadataFilter(),
            ImageTransform(),
            FinebooruWriter()
        ]
    ).run(limit=1, workers=1)

