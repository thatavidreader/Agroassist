from flask import Flask, request, jsonify,  render_template, send_file
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from flask_cors import CORS
import requests
import numpy as np
import folium
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, BBox
from tensorflow.keras.models import load_model
import pickle
import io

app = Flask(__name__)
CORS(app)  # This allows React to send requests to Flask


################## DISEASE DETECTION STUFF BELOW ######################################

# Load the model and feature extractor
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Get class labels from model config
id2label = model.config.id2label  # Dictionary: {0: "Healthy", 1: "Powdery Mildew", ...}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = Image.open(file).convert("RGB")
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make a prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = predictions.argmax().item()

    # Get class name
    predicted_class_name = id2label.get(predicted_class_idx, "Unknown")

    # Cure List 
    # Define Cures for Each Disease
    disease_cures = {
    # ================== TOMATO DISEASES ==================
    "Tomato Bacterial Spot": "Remove infected plants and apply copper-based sprays like Kocide 3000.",
    "Tomato Early Blight": "Apply chlorothalonil fungicides and remove lower infected leaves.",
    "Tomato Late Blight": "Destroy infected plants immediately and use fungicides like Revus.",
    "Tomato Leaf Mold": "Improve air circulation and spray potassium bicarbonate solutions.",
    "Tomato Septoria Leaf Spot": "Remove affected leaves and apply mancozeb fungicides.",
    "Tomato Spider Mites": "Spray with insecticidal soap or neem oil every 5-7 days.",
    "Tomato Target Spot": "Use azoxystrobin fungicides and avoid overhead watering.",
    "Tomato Yellow Leaf Curl": "Control whiteflies with imidacloprid and remove infected plants.",
    "Tomato Mosaic Virus": "Destroy infected plants and disinfect tools with bleach solution.",
    "Tomato Healthy": "No treatment needed. Continue regular monitoring.",

    # ================== POTATO DISEASES ==================
    "Potato Early Blight": "Apply copper fungicides weekly and practice crop rotation.",
    "Potato Late Blight": "Use preventative fungicides like Tanos before infection occurs.",
    "Potato Healthy": "Maintain proper soil moisture and fertilization.",

    # ================== APPLE DISEASES ==================
    "Apple Scab": "Remove infected leaves and apply fungicides like captan or myclobutanil.",
    "Apple Black Rot": "Prune affected branches and use copper-based fungicides.",
    "Apple Cedar Rust": "Remove nearby cedar trees and apply mancozeb fungicides.",
    "Apple Healthy": "No action needed. Maintain proper care.",

    # ================== CORN DISEASES ==================
    "Corn Common Rust": "Plant resistant hybrids and apply sulfur dust.",
    "Corn Gray Leaf Spot": "Rotate crops and use azoxystrobin fungicides.",
    "Corn Northern Leaf Blight": "Apply tebuconazole fungicides early in season.",
    "Corn Healthy": "Continue normal growing practices.",

    # ================== GRAPE DISEASES ==================
    "Grape Black Rot": "Remove mummified fruit and apply mancozeb sprays.",
    "Grape Esca": "Prune infected wood and protect pruning wounds.",
    "Grape Healthy": "Monitor regularly and maintain canopy airflow.",

    # ================== STRAWBERRY DISEASES ==================
    "Strawberry Leaf Scorch": "Remove old leaves and apply thiophanate-methyl.",
    "Strawberry Healthy": "Renovate beds annually for continued health.",

    # ================== PEPPER DISEASES ==================
    "Pepper Bacterial Spot": "Use streptomycin sprays and disease-free seeds.",
    "Pepper Healthy": "Rotate crops every 2-3 years.",

    # ================== CHERRY DISEASES ==================
    "Cherry Powdery Mildew": "Apply sulfur sprays at first sign of infection.",
    "Cherry Healthy": "Prune for sunlight penetration and airflow.",

    # ================== OTHER CROPS ==================
    "Squash Powdery Mildew": "Spray with potassium bicarbonate solutions weekly.",
    "Soybean Healthy": "No treatment required for healthy plants.",
    "Raspberry Healthy": "Maintain proper spacing between plants."
}




    # Get Cure :

    predicted_cure = disease_cures.get(predicted_class_name)


    print({'predicted_class': predicted_class_name,
           'predicted_cure': predicted_cure})
    return jsonify({'predicted_class': predicted_class_name,
           'predicted_cure': predicted_cure})

##########################################################################################

######################## IRRIGATION STUFF BELOW ##########################################
# Sentinel Hub Configuration
config = SHConfig()
config.sh_client_id = "cbdf46c0-c676-4218-9f2e-fb4efc3109da"
config.sh_client_secret = "FEDSA1s13BjmlhgAkcgpuedLkwAlUZI9"

# NDMI Calculation Script
evalscript_ndmi = """
function setup() {
    return {
        input: ["B08", "B11"],
        output: { bands: 1, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11);
    return [ndmi];
}
"""

@app.route("/get_ndmi", methods=["GET"])
def get_ndmi():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    
    if lat is None or lon is None:
        return {"error": "Latitude and Longitude are required."}, 400
    
    bbox_size = 0.005  # 500m area
    bbox_coords = [
        lon - bbox_size / 2, lat - bbox_size / 2,
        lon + bbox_size / 2, lat + bbox_size / 2
    ]
    bbox = BBox(bbox=bbox_coords, crs="EPSG:4326")
    size = bbox_to_dimensions(bbox, resolution=10)  # 10m per pixel

    # Fetch NDMI Data
    ndmi_request = SentinelHubRequest(
        evalscript=evalscript_ndmi,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2024-09-01", "2024-09-29"),
            maxcc=0.2
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config
    )
    ndmi_image = ndmi_request.get_data()[0].astype(np.float32)

    # Define NDMI Thresholds
    ndmi_threshold_red = 0.1  # Dry areas
    ndmi_threshold_yellow = 0.15  # Moderate dryness

    # Identify dry pixels
    dry_pixels_red = np.where(ndmi_image < ndmi_threshold_red)
    dry_pixels_yellow = np.where((ndmi_image >= ndmi_threshold_red) & (ndmi_image < ndmi_threshold_yellow))

    # Convert pixels to lat/lon
    def pixel_to_geo(pixel_x, pixel_y, bbox, img_size):
        lon_min, lat_min, lon_max, lat_max = bbox_coords
        img_width, img_height = img_size
        lon = lon_min + (pixel_x / img_width) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / img_height) * (lat_max - lat_min)
        return [lat, lon]

    dry_locations_red = [pixel_to_geo(x, y, bbox, size) for x, y in zip(dry_pixels_red[1], dry_pixels_red[0])]
    dry_locations_yellow = [pixel_to_geo(x, y, bbox, size) for x, y in zip(dry_pixels_yellow[1], dry_pixels_yellow[0])]

    # Generate Map
    farm_map = folium.Map(location=[lat, lon], zoom_start=15, tiles="Esri WorldImagery")

    for lat, lon in dry_locations_red:
        folium.CircleMarker(location=[lat, lon], radius=3, color='red', fill=True, fill_color='red').add_to(farm_map)
    
    for lat, lon in dry_locations_yellow:
        folium.CircleMarker(location=[lat, lon], radius=3, color='yellow', fill=True, fill_color='yellow').add_to(farm_map)

    # Save and return HTML map
    map_path = "static/dry_areas_map.html"
    farm_map.save(map_path)
    return send_file(map_path, mimetype='text/html')

#######################################################################################################

########################################### CROP RECOMENDATION ########################################

crop_model = pickle.load(open('RandomForest.pkl', 'rb'))
soil_model = pickle.load(open('model.pkl', 'rb'))

SOIL_PROFILES = {
    'black':   {'N': 65, 'P': 40, 'K': 55, 'ph': 6.8},
    'cinder':  {'N': 30, 'P': 15, 'K': 25, 'ph': 5.2},
    'laterite':{'N': 45, 'P': 20, 'K': 30, 'ph': 5.8},
    'peat':    {'N': 80, 'P': 35, 'K': 40, 'ph': 4.5},
    'yellow':  {'N': 35, 'P': 25, 'K': 35, 'ph': 6.0}
}

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    img_file = request.files['image']
    img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = soil_model.predict(img_array)
    soil_class = ['black', 'cinder', 'laterite', 'peat', 'yellow'][np.argmax(pred)]
    soil_params = SOIL_PROFILES[soil_class]
    
    input_features = [
        soil_params['N'],
        soil_params['P'],
        soil_params['K'],
        26,   # Temperature
        75,   # Humidity
        soil_params['ph'],
        150   # Rainfall
    ]
    
    crop_probs = crop_model.predict_proba([input_features])[0]
    recommendations = sorted(zip(crop_model.classes_, crop_probs), key=lambda x: -x[1])[:3]
    
    response = {
        'soil_type': soil_class,
        'recommendations': [
            {'crop': crop, 'probability': float(prob)}
            for crop, prob in recommendations
        ]
    }

    print(response)
    
    return jsonify(response)

#######################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
