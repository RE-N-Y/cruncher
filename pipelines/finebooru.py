import tarfile
from pathlib import Path
from base import Reader, Step, FilterStep, Batch
from huggingface_hub import HfFileSystem
from s3fs import S3FileSystem
from fsspec.implementations.dirfs import DirFileSystem

import os,io
import jsonpickle
import traceback
import duckdb
from loguru import logger
from PIL import Image
from typing import Optional
from components.images.base import ImageData
from utils import timer
from tqdm import tqdm

class FinebooruReader(Reader):
    name = "FinebooruReader"
    src = "datasets/nyanko7/danbooru2023"

    def __init__(self, wkdir:Path):
        self.hffs = HfFileSystem()
        self.wkdir = wkdir
        self.dirfs = DirFileSystem(wkdir)
        
        self.load()

    @timer()
    def load(self):
        self.db = duckdb.connect(":memory:")

        logger.info("Downloading posts.tar.gz from HF")
        targz = self.wkdir/"posts.tar.gz"
        if not targz.exists():
            with self.hffs.open(self.src + "/" + "metadata/posts.tar.gz", "rb") as f:
                with self.dirfs.open(str(targz), "wb") as g:
                    g.write(f.read())

        dotjson = self.wkdir/"posts.json"
        if not dotjson.exists():
            with tarfile.open(str(targz), "r:gz") as tar:
                tar.extract("posts.json", path=self.wkdir)

        self.db.execute("CREATE TABLE posts AS SELECT * FROM read_json_auto('posts.json')")
        logger.info("posts metadata loaded into duckdb")

    @timer()
    def scheduler(self) -> list[dict]:
        return self.hffs.ls(self.src + "/" + "original")

    @timer()
    def read(self, file:dict) -> Batch[list[ImageData]]:
        logger.info(f"Reading {file['name']}")
        file = Path(file["name"])

        logger.info(f"Extracting {file}")
        tar = self.wkdir/file.name
        if not tar.exists():
            with self.hffs.open(file, "rb") as f:
                with self.dirfs.open(str(tar), "wb") as g:
                    g.write(f.read())

        with tarfile.open(str(tar)) as t:
            images = t.getmembers()
        
        tarfolder = self.wkdir/file.stem
        # check if tar is already extracted
        if not tarfolder.exists():
            with tarfile.open(str(tar)) as t:
                t.extractall(tarfolder)

        logger.info(f"Extracted {file} to {tarfolder}")
        batch = Batch(content=[], metadata=None)

        for image in tqdm(images):
            try:
                imagename = Path(image.name)
                image = Image.open(tarfolder/imagename)
                metadata = self.db.query(f"SELECT * FROM posts WHERE id = {imagename.stem}")
                metadata = metadata.df().to_dict(orient="records")[0]
                metadata["file_size"] = os.path.getsize(tarfolder/imagename)
                
                data = ImageData(image=image, metadata=metadata)
                batch.content.append(data)
            except Exception as e:
                logger.error(f"Error reading {imagename}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        return batch

class FinebooruMetadataFilter(FilterStep):
    name = "FinebooruMetadataFilter"
    version = "0.1"
    minheight = 256
    minwidth = 256
    minsize = 10 # in KB

    @timer(verbosity=0)
    def step(self, data:ImageData) -> Optional[ImageData]:
        expected = set(["id", "rating", "score", "tag_string_general", "tag_count_general", "image_height", "image_width"])
        actual = set(data.metadata.keys())

        if not expected.issubset(actual):
            return None
        # rating must be s, g, or q
        if data.metadata["rating"] != "s" and data.metadata["rating"] != "g" and data.metadata["rating"] != "q":
            return None
        # discard images with negative score
        if data.metadata["score"] < 0:
            return None
        # discard images with tag count less than 4
        if data.metadata["tag_count_general"] < 4:
            return None
        
        if data.metadata["image_height"] < self.minheight or data.metadata["image_width"] < self.minwidth:
            return None
        
        if data.metadata["file_size"] < self.minsize * 1024:
            return None

        return data
    
class FinebooruImageSizeFilter(FilterStep):
    name = "FinebooruImageSizeFilter"
    version = "0.1"
    min_size = 512

    @timer(verbosity=0)
    def step(self, data:ImageData) -> Optional[ImageData]:
        width, height = data.image.size
        if width < self.min_size or height < self.min_size:
            return None
        return data
    
class FinebooruWriter(Step):
    name = "FinebooruWriter"
    version = "0.1"

    def __init__(self):
        self.tgt = "finebooru/raw"
        self.tgtfs = S3FileSystem(
            key="ZXNA022VQU5EH2CN6QTT",
            secret="lWPIXlJgBrljobeMP7127hLvYTlGOw0bChw7IgYe",
            endpoint_url="https://s3.us-east-1.wasabisys.com"
        )

    @timer(verbosity=0)
    def step(self, data:ImageData):
        id = data.metadata["id"]
        tgt = self.tgt + "/" + str(id) + ".jpg"
        metatgt = self.tgt + "/" + str(id) + ".json"

        with self.tgtfs.open(tgt, "wb") as f:
            buffer = io.BytesIO()
            data.image.save(buffer, format="JPEG")
            f.write(buffer.getvalue())
        
        with self.tgtfs.open(metatgt, "w") as f:
            f.write(jsonpickle.encode(data.metadata))
        






        
                