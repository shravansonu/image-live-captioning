from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.eval()

def generate_caption_pretrained(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=30,
            num_beams=5
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# Test
if __name__ == "__main__":
    img_path = "test.jpeg"  # change image
    print("Pretrained Caption:")
    print(generate_caption_pretrained(img_path))
