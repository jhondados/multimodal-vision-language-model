"""Multimodal vision-language pipeline."""
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io, base64, json

class VisionLanguagePipeline:
    def __init__(self, project_id: str):
        vertexai.init(project=project_id, location="us-central1")
        self.model = GenerativeModel("gemini-1.5-pro-002")

    def chart_to_insights(self, image_bytes: bytes) -> dict:
        img = Part.from_data(image_bytes, "image/png")
        prompt = """Analyze this chart/graph and return JSON with:
        - chart_type, title, x_axis, y_axis
        - key_values: list of {label, value}
        - trend: overall trend description
        - top_insight: most important business insight
        - anomalies: any unusual patterns"""
        r = self.model.generate_content([img, prompt],
            generation_config={"response_mime_type": "application/json"})
        return json.loads(r.text)

    def document_vqa(self, image_bytes: bytes, question: str) -> str:
        img = Part.from_data(image_bytes, "image/png")
        return self.model.generate_content([img, f"Answer precisely: {question}"]).text

    def video_qa(self, gcs_video_uri: str, question: str) -> str:
        video = Part.from_uri(gcs_video_uri, "video/mp4")
        return self.model.generate_content([video, question]).text
