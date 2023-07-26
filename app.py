from flask import Flask, request, render_template, jsonify
from models import GeneratorRRDB
from datasets import denormalize
import torch
from torch.autograd import Variable
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model
opt = {
    "image_path": None,
    "checkpoint_model": "generator_final.pth",  # Replace with the actual path
    "channels": 3,
    "residual_blocks": 23,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = GeneratorRRDB(opt["channels"], filters=64, num_res_blocks=opt["residual_blocks"]).to(device)
checkpoint = torch.load(opt["checkpoint_model"], map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint)
generator.eval()

# Define the mean and std for RGB images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def super_resolution(image):
    image_tensor = Variable(transform(image)).to(device).unsqueeze(0)
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()
    return sr_image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        
        if file:
            try:
                pil_image = Image.open(file.stream).convert("RGB")
                sr_image = super_resolution(pil_image)
                
                # Save the SR image to a BytesIO buffer
                output_buffer = BytesIO()
                save_image(sr_image, output_buffer, format="PNG")
                output_buffer.seek(0)
                
                # Return the SR image as response
                return output_buffer.getvalue(), 200, {'Content-Type': 'image/png'}
            except Exception as e:
                return jsonify({"error": str(e)})
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)