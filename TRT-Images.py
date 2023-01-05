from roboflow import Roboflow

local_inference_server_address = "http://localhost:9001/"
version_number = 8

rf = Roboflow(api_key="API")
project = rf.workspace().project("small-test-set")
local_model = project.version(version_number=version_number, local=local_inference_server_address).model

# infer on a local image
inference = local_model.predict("Osprey.jpg", confidence=40, overlap=30).json

# print full inference
print(inference)

