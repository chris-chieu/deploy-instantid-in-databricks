# Databricks notebook source
!pip install --upgrade diffusers

!pip install opencv-python
!pip install insightface
!pip install onnxruntime-gpu

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow

class InstantIDImgToImg(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.pipe = None
        self.app = None


    def convert_from_image_to_cv2(self, img):
        import cv2
        import numpy as np
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def resize_img(self, 
        input_image,
        target_size=1024,
    ):
        import PIL 
        from PIL import Image
        w, h = input_image.size

        # Calculate the scaling factor to fit the image within the target size
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image while maintaining aspect ratio
        resized_image = input_image.resize((new_w, new_h), Image.BILINEAR)

        # Create a blank white canvas of target size
        input_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))

        # Paste the resized image onto the center of the canvas
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        input_image.paste(resized_image, (paste_x, paste_y))

        return input_image
      
    def image_to_base64(self, image):
        from io import BytesIO
        import base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8') 

    def base64_to_image(self, base64_string):
        from io import BytesIO
        import base64
        import PIL 
        from PIL import Image
    # Decode the base64 string
        img_data = base64.b64decode(base64_string)
    
    # Create a BytesIO object from the decoded data
        buffer = BytesIO(img_data)
    
    # Open the image using PIL
        image = Image.open(buffer)
    
        return image
    
    def load_context(self, context):
        import diffusers
        from diffusers.models import ControlNetModel
        import importlib.util
        import sys

        import os
        import shutil
        import torch
        
        from insightface.app import FaceAnalysis
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        
        

            # If loading for the first time, there will be an AssertionError. This will be fixed in the next cell.
            # Load face encoder
            
        try:
            app = FaceAnalysis(
            name="antelopev2",
            root="./artifacts",
            providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'],
        )
            app.prepare(ctx_id=0, det_size=(640, 640))
            
        except:
            source_path = "./artifacts/models/antelopev2/antelopev2"
            destination_path = "./artifacts/models/antelopev2"

            # List files
            if os.path.exists(source_path):
                files = os.listdir(source_path)
            else:
                files = []

            # Copy directory contents
            if os.path.isdir(source_path) and not os.path.exists(destination_path):
                shutil.copytree(source_path, destination_path)
            elif os.path.isdir(source_path):  # For safety if destination exists
                for file in os.listdir(source_path):
                    shutil.copy2(os.path.join(source_path, file), destination_path)

            # Remove directory
            if os.path.exists(source_path):
                shutil.rmtree(source_path)
#
                # Load face encoder again
                self.app = FaceAnalysis(
                name="antelopev2",
                root="./artifacts",
                providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'],
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
            
        # prepare models under ./checkpoints
  

        face_adapter = context.artifacts["face_adapter"]
        controlnet_path = context.artifacts["controlnet"]

        # load IdentityNet
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        self.pipe.cuda()
        # load adapter
        self.pipe.load_ip_adapter_instantid(face_adapter)


    def predict(self, context, model_input):
        from pipeline_stable_diffusion_xl_instantid import draw_kps
        
        prompt = model_input["prompt"][0]

        #negative_prompt = model_input["negative_prompt"][0]

        #controlnet_conditioning_scale = model_input["controlnet_conditioning_scale"][0]

        #ip_adapter_scale = model_input["ip_adapter_scale"][0]

        init_image = self.base64_to_image(model_input["init_image"][0])

        face_image = self.resize_img(init_image, target_size=1024)
    
    
        face_image_cv2 = self.convert_from_image_to_cv2(face_image)
    
        face_info = self.app.get(face_image_cv2)
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use   the       maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])
    
        # prompt
        #prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome,   high      contrast, dramatic shadows, 1940s style, mysterious, cinematic"
        negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
    
        #prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage,     Kodachrome,       Lomography, stained, highly detailed, found footage, masterpiece, best quality"
        #negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing,    illustration,      glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2),   (text:1.2), watermark,      painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
    
    
        # generate image
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
        ).images[0]
    
        return self.image_to_base64(image)



# COMMAND ----------

# DBTITLE 1,Register the model in uc
mlflow.set_registry_uri('databricks-uc')

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec
import pandas as pd
from PIL import Image
from io import BytesIO
import base64


def load_image_from_volume(volume_path):
      import PIL 
      from PIL import Image
      with Image.open(volume_path) as img:
        return img.convert("RGB")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


input_schema = Schema([ColSpec(DataType.string, "prompt"),
                       ColSpec(DataType.string, "init_image")
                       ])

input_schema = Schema([ColSpec(DataType.string, "prompt"),
                       ColSpec(DataType.string, "init_image")])

output_schema=Schema([ColSpec(DataType.string, "image")])

signature = ModelSignature(inputs=input_schema,outputs=output_schema)


image = image_to_base64(load_image_from_volume("/Volumes/catalog/schema/volume/image.jpeg"))

prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

# Define input example
input_example=pd.DataFrame({"prompt" : [prompt],
              "init_image" : [image]
              })


# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=InstantIDImgToImg(),
        input_example=input_example,
        code_paths=["./pipeline_stable_diffusion_xl_instantid.py", "./ip_adapter"],
        signature=signature,
        registered_model_name="catalog.schema.instantid-model",
        artifacts={
                   "face_adapter": "/Volumes/catalog/schema/volume/checkpoints/ip-adapter.bin",
                  "controlnet": "/Volumes/catalog/schema/volume/checkpoints/ControlNetModel",
                   },
        pip_requirements=["transformers==4.48.0", "torch==2.5.1", "torchvision==0.20.1" , "accelerate", "diffusers==0.32.2", "huggingface_hub==0.27.1", "invisible-watermark>=0.2.0", "bitsandbytes==0.45.4", "sentencepiece==0.2.0", "insightface==0.7.3", "onnxruntime-gpu==1.22.0"]
    )
    
    

# COMMAND ----------

# DBTITLE 1,Set tag in production
from mlflow import MlflowClient

client = MlflowClient()

model_name = "catalog.schema.instantid-model"

# Get latest version (UC-compatible)
latest_version = client.search_model_versions(f"name='{model_name}'")[0].version

alias_name = "production"

client.set_model_version_tag(
    name="catalog.schema.instantid-model",
    version=latest_version,
    key="RemoveAfter",
    value="20251231"
)



client.set_registered_model_alias(model_name, alias_name, latest_version)




# COMMAND ----------

# DBTITLE 1,Serve the model
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

entity_version = client.search_model_versions(f"name='{model_name}'")[0].version

# Create a serving endpoint using the alias
endpoint_name = "instantid-endpoint"
model_uri = f"models:/{model_name}@{alias_name}"  # Use alias in URI

model_config = ServedEntityInput(
    entity_name=model_name,
                entity_version=entity_version,  # Automatically resolves to alias target
                workload_type="GPU_MEDIUM",
                workload_size="Small",
                scale_to_zero_enabled=True
)

try:
    w.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
      name=endpoint_name,
        served_entities=[model_config]
    )
)
except:
    w.serving_endpoints.update_config(
    name=endpoint_name,  # Existing endpoint name
    served_entities=[model_config]  # Replace with new model version
)

##### You will have to wait for approximately 20-25 minutes before the endpoint is ready
print(f"Serving endpoint '{endpoint_name}' created using alias '{entity_version}'.")

# COMMAND ----------

