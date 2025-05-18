import onnxruntime as ort
import cv2
import numpy as np

PARAM_NANOTRACK_BACKBONE_PATH = "nanotrack_backbone_path"
PARAM_NANOTRACK_HEAD_PATH = "nanotrack_head_path"

backbone_session = ort.InferenceSession("/workspace/src/tracker_nano/tracker_nano/models/nanotrack_backbone_sim.onnx")
head_session     = ort.InferenceSession("/workspace/src/tracker_nano/tracker_nano/models/nanotrack_head_sim.onnx")

def preprocess(img, size):
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...]
    return img

template_img = preprocess(cv2.imread("/workspace/src/tracker_tester/data/template.jpg"), (255, 255))
search_img   = preprocess(cv2.imread("/workspace/src/tracker_tester/data/search.jpg"), (255, 255))
print(template_img.shape)
# Run template through backbone
feat_z = backbone_session.run(None, {backbone_session.get_inputs()[0].name: template_img})[0]

# Run search through backbone
feat_x = backbone_session.run(None, {backbone_session.get_inputs()[0].name: search_img})[0]

# Prepare head inputs: [template_feature, search_feature]
head_inputs = {
    head_session.get_inputs()[0].name: feat_z,
    head_session.get_inputs()[1].name: feat_x
}

print(head_inputs)
cls_output, loc_output = head_session.run(None, head_inputs)

print("cls:", cls_output.shape)
print("loc:", loc_output.shape)
