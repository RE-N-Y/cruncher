from base import Step, Batch, FilterStep
from pathlib import Path
from io import BytesIO

from PIL import Image
from PIL.Image import Image as PILImage

import numpy as np
import onnxruntime as ort
from imagehash import phash, whash
from dataclasses import dataclass
from einops import rearrange

import torch
from transformers import pipeline
from transformers import AutoImageProcessor, ViTForImageClassification

@dataclass
class ImageData:
    image:PILImage
    metadata:dict
    
@dataclass
class UpscalerConfig:
    model:str
    dtype:np.float16 | np.float32
    scale:int

class AnimeUpscaler(Step):
    name = "AnimeUpscaler"

    def __init__(self, model="esrgan"):
        self.config:UpscalerConfig = self.get_model_config(model)
        self.session = ort.InferenceSession(self.config.model, providers=[['CUDAExecutionProvider', 'CPUExecutionProvider']])
        
    def get_model_config(model:str) -> UpscalerConfig:
        match model:
            case "esrgan":
                return UpscalerConfig(model="models/esrgan/RealESRGAN_Anime_6B.onnx", dtype=np.float16, scale=4)            
            case "animejanai-v3-compact-fp16":
                return UpscalerConfig(model="models/animejanai/2x_AnimeJaNai_HD_V3_Compact_583k-fp16.onnx", dtype=np.float16, scale=2)
            case "animejanai-v3-ultra-compact-fp16":
                return UpscalerConfig(model="models/animejanai/2x_AnimeJaNai_HD_V3_UltraCompact_425k-fp16.onnx", dtype=np.float16, scale=2)
            case "animejanai-v3-super-ultra-compact-fp16":
                return UpscalerConfig(model="models/animejanai/2x_AnimeJaNai_HD_V3_SuperUltraCompact_5k-fp16.onnx", dtype=np.float16, scale=2)
            case _:
                raise ValueError(f"Unknown upscaler model {model}")

    def step(self, data:ImageData) -> ImageData:
        image = data.image.convert('RGB')
        
        tensor = np.asarray(image, dtype=self.config.dtype)
        tensor = rearrange(tensor / 255, 'h w c -> 1 c h w')
        [output] = self.session.run(None, {'input':tensor})
        output = rearrange(output, '1 c h w -> h w c')
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        # update image data
        data.image = Image.fromarray(output)
        data.metadata['upscaled'] = True
        data.metadata['image_width'], data.metadata['image_height'] = data.image.size

        return data

        

class ImageTransform(Step):
    name = "ImageTransform"

    def __init__(
        self, 
        max_height:int = 4096,
        max_width:int = 4096,
        multiple:int = 16,
        resize_sampling:Image.Resampling = Image.LANCZOS,
        quality:int = 95,
        format:str="RGB"
    ):
        self.max_height = max_height
        self.max_width = max_width
        self.multiple = multiple
        self.resize_sampling = resize_sampling
        self.quality = quality
        self.format = format


    def step(self, data:ImageData) -> ImageData:
        img = data.image

        img.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
        width, height = img.size
        height, width = height - height % self.multiple, width - width % self.multiple
        img = img.crop((0, 0, width, height))
        img = img.convert(self.format)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        img = Image.open(buffer)

        # update image data
        data.image = img
        data.metadata['image_width'], data.metadata['image_height'] = img.size
        data.metadata["file_size"] = buffer.getbuffer().nbytes

        return data
    


class Deduplicate(Step):
    name = "Deduplicate"

    def __init__(self, hashfunc="phash"):
        match hashfunc:
            case "phash":
                self.hashfunc = phash
            case "whash":
                self.hashfunc = whash
            case _:
                raise ValueError(f"Unknown hash function {hashfunc}")


    def run(self, batch:Batch) -> Batch:
        cache = set()
        deduped:list[ImageData] = []
        data:list[ImageData] = batch.content

        for d in data:
            ihash = self.hashfunc(d.image)
            if ihash in cache:
                continue
            cache.add(ihash)
            deduped.append(d)

        return deduped


        
        
class QualityFilter(FilterStep):
    name = "QualityFilter"
    tag = "shadowlilac/aesthetic-shadow-v2"

    def __init__(self, threshold, dtype=torch.bfloat16):
        self.threshold = threshold
        self.processor = AutoImageProcessor.from_pretrained(self.tag)
        self.model = ViTForImageClassification.from_pretrained(
            self.tag, torch_dtype=dtype, device_map="auto", 
            low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation="sdpa",
        )
        
    @torch.inference_mode()
    def batch(self, batch:Batch) -> Batch:
        inputs = np.array([d.image for d in batch.content])
        inputs = self.processor(inputs, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        
        for d, score in zip(batch.content, scores):
            d.metadata['score'] = score
        return [d for d, score in zip(batch.content, scores) if score > self.threshold]