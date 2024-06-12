import tarfile
from pathlib import Path
from base import Reader, Step, FilterStep, Batch
from huggingface_hub import HfFileSystem
from s3fs import S3FileSystem
from fsspec.implementations.dirfs import DirFileSystem

import os,io
from json import JSONEncoder
from uuid import UUID
import traceback
import duckdb
from loguru import logger
from PIL import Image
from typing import Optional
from components.images.base import ImageData
from utils import timer
import webdataset as wds
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tarfile import TarInfo

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
    
    def _read(self, image:TarInfo, tarfolder:Path) -> Optional[ImageData]:
        try:
            imagename = Path(image.name)
            image = Image.open(tarfolder/imagename)
            metadata = self.db.query(f"SELECT * FROM posts WHERE id = {imagename.stem}")
            metadata = metadata.df().to_dict(orient="records")[0]
            metadata["file_size"] = os.path.getsize(tarfolder/imagename)
            data = ImageData(image=image, metadata=metadata)
            return data
        except Exception as e:
            logger.error(f"Error reading {imagename}: {e}")
            logger.error(traceback.format_exc())
            return None

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

        batch.content = Parallel()(delayed(self._read)(image, tarfolder) for image in tqdm(images))
        batch.content = [d for d in batch.content if d is not None]

        # tarfolder = "data-0042"
        tarfolder = str(tarfolder)
        # webdataset requires each .tar shard to be of 000000.tar format
        batch.metadata = { "key": tarfolder.split("-")[-1].zfill(6) }

        return batch

class FinebooruMetadataFilter(FilterStep):
    name = "FinebooruMetadataFilter"
    version = "0.1"
    minheight = 256
    minwidth = 256
    minsize = 10 * 1024
    blacklist= [ 
        "translated", "translation", "comic", "speech_bubble", "4koma", "english_text", "doujinshi",
        "furry", "sketch", "cosplay", "realistic", "zoom_layer", "scan",
        "3d", "photo_(medium)", "oil_painting_(medium)", "painting_(medium)", "unfinished", "traditional_media", "watermark",
    ]

    @timer(verbosity=0)
    def step(self, data:ImageData) -> Optional[ImageData]:
        expected = set(["id", "rating", "score", "tag_string", "tag_string_general", "tag_count_general", "image_height", "image_width"])
        actual = set(data.metadata.keys())

        if not expected.issubset(actual):
            logger.info(f"Missing metadata for {data.metadata['id']}")
            return None
        # rating must be s, g, or q
        if data.metadata["rating"] != "s" and data.metadata["rating"] != "g" and data.metadata["rating"] != "q":
            return None
        # discard images with negative score
        if data.metadata["score"] < 0:
            logger.info(f"Negative score for {data.metadata['id']}")
            return None
        # discard images with tag count less than 4
        if data.metadata["tag_count_general"] < 4:
            logger.info(f"Tag count less than 4 for {data.metadata['id']}")
            return None
        
        if data.metadata["image_height"] < self.minheight or data.metadata["image_width"] < self.minwidth:
            logger.info(f"Image size less than {self.minheight}x{self.minwidth} for {data.metadata['id']}")
            return None
        
        if data.metadata["file_size"] < self.minsize:
            logger.info(f"File size less than {self.minsize} for {data.metadata['id']}")
            return None
        
        for tag in self.blacklist:
            if tag in data.metadata["tag_string"]:
                logger.info(f"Blacklisted tag {tag} for {data.metadata['id']}")
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
        self.s3 = S3FileSystem(
            key="ZXNA022VQU5EH2CN6QTT",
            secret="lWPIXlJgBrljobeMP7127hLvYTlGOw0bChw7IgYe",
            endpoint_url="https://s3.us-east-1.wasabisys.com"
        )

    def encodable(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, list):
            return [self.encodable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self.encodable(v) for k, v in obj.items()}
        else:
            return obj
        

    @timer()
    def run(self, batch:Batch[ImageData]):
        tar = "data-" + batch.metadata["key"] + ".tar"
        logger.info(f"Writing shard {tar}")
        with wds.TarWriter(tar) as sink:
            for data in tqdm(batch.content):
                sink.write({
                    "__key__": str(data.metadata["id"]),
                    "jpg": data.image,
                    "json": self.encodable(data.metadata)
                })

        logger.info(f"Uploading {tar} to {self.tgt}")
        with open(tar, "rb") as f:
            with self.s3.open(f"{self.tgt}/{tar}", "wb") as g:
                g.write(f.read())
        logger.info(f"Uploaded {tar} to {self.tgt}")
        
        






        
                