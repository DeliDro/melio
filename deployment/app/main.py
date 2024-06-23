import re
import argparse
from typing import List

from nllb import Translater

from kserve.utils.utils import generate_uuid
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse


CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]]'


def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()


class MyModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        
        self.name: str = name
        self.model: Translater = None
        self.ready: bool = False
        
        self.load()

    def load(self):
        # Instantiate model
        self.model = Translater()
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> List[str]:
        # Get data from payload
        print(f"[[INPUT]] => {payload.inputs[0].data}")
        return payload.inputs[0].data
        
        # infer_inputs: List[str] = payload.inputs[0].data
        # print(f"** infer_input ({type(infer_inputs)}): {infer_inputs}")

        # cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        # print(f"** cleaned_text ({type(cleaned_texts)}): {cleaned_texts}")
        # return cleaned_texts

    def predict(self, data: List[str], *args, **kwargs) -> InferResponse:
        response_id = generate_uuid()
        results: List[str] = [self.model.predict(sentence) for sentence in data]
        # print(f"** result ({type(results)}): {results}")
        print(f"[[PREDICTION]] => {results}")

        infer_output = InferOutput(name="output-0", shape=[len(results)], datatype="STR", data=results)
        
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        
        return infer_response

parser = argparse.ArgumentParser(parents=[model_server.parser])

parser.add_argument(
    "--model_name",
    default="model",
    help="The name that the model is served under."
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    # model = MyModel("model")
    server = ModelServer()
    server.start([model])

    # print(Translater().predict("ne ye ameriken ye"))