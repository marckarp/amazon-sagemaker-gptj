# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
import json
import os
from pathlib import Path
import flask
import logging
import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)
import deepspeed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Inference Endpoint')

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(deepspeed.__version__)

class ScoringService(object):
    generator = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.generator is None:
            # Determine device -- needs to be updated for multi gpu

            model_path = "/opt/ml/model/"
            print(f"Downloading model located at: {os.environ['S3_MODEL_LOCATION']}")
            os.system(f"aws s3 cp {os.environ['S3_MODEL_LOCATION']} /opt/ml/model/ --recursive")

            logger.info('Loading model ...')
            
            model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model = deepspeed.init_inference(model,
                                        mp_size=1,
                                        dtype=torch.half,
                                        replace_method='auto',
                                        replace_with_kernel_inject=True
                                                      )
            generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=local_rank,torch_dtype=torch.float16)
            
            torch.cuda.synchronize()

            cls.generator = generator

            return cls.generator

        return cls.generator

    @classmethod
    def predict(cls, inputs):

        gen = cls.get_model()
        
        parameters = inputs.pop('parameters')
        
        #generator(input_text, do_sample=True, max_length=2047,min_length=2047, top_k=50, top_p=0.95, temperature=0.9)
        return gen(**inputs, **parameters)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single input.
    """

    # data should be a dictionary with an 'inputs' key
    data = flask.request.get_json()
    if isinstance(data, str):
        data = json.loads(data)

    inputs = data['inputs']

    predictions = ScoringService.predict(inputs)

    return flask.jsonify(response=predictions, status=200)
