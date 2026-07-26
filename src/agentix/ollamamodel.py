""" Create an Ollama client and send messages to a local model """

import logging
from tqdm import tqdm
from ollama import Client

logger = logging.getLogger(__name__)

def response2message(response):
    """
    Extracts and processes the `content` field from the `message` object
    within the given response if certain conditions are satisfied.
    """
    output = None
    if response.get('done', None):
        if response.get('done_reason', None) == 'stop':
            output = response.message.content
            # clean up the response message
            output = output.\
                replace('\n', '').\
                replace('\t', '').\
                replace(' ','')
    return output

class Ollama:
    """
    Class for managing interactions with the Ollama server, including managing
    models, sending messages, and receiving responses.

    This class is designed to handle client creation, model management, and
    interaction with the Ollama server using provided endpoints. It automates
    tasks such as pulling models, listing available models, and sending messages
    with specified parameters.

    :ivar host: The hostname of the Ollama server.
    :type host: str
    :ivar port: The port used to connect to the Ollama server.
    :type port: int
    :ivar models: List of models available on the Ollama server.
    :type models: list
    """
    def __init__(self, host: str = 'ollama', port: int = 11434):
        self.host = host
        self.port = port
        self.models = self.list_models()

    def create_client(self):
        client = None
        url = f'http://{self.host}:{self.port}'
        try:
            client = Client(host=url)
        except Exception as e:
            logger.error(f'Error creating client: {e}')
        return client

    def list_models(self):
        client = self.create_client()
        model_list = None
        try:
            model_list = client.list().get('models', None)
            model_list = [model.get('model', None) for model in model_list]
        except Exception as e:
            logger.error(f'Error retrieving model list: {e}')
        return model_list

    def pull_model(self, model_name: str) -> bool:
        """
        Pull (download) a model from the Ollama server with a progress bar.
        Args:
            model_name (str): Name of the model to pull
        Returns:
            bool: True if successful, False otherwise
        """
        client = self.create_client()
        output = False
        if not client:
            logger.error(f'Failed to create client connection')
        try:
            # Start pulling the model and listen to progress
            with tqdm(unit='B', unit_scale=True, desc=model_name, dynamic_ncols=True) as pbar:
                for progress in client.pull(model_name, stream=True):
                    # Each progress update is a dictionary
                    if 'total' in progress and 'completed' in progress:
                        pbar.total = progress['total']
                        pbar.update(progress['completed'] - pbar.n)
                    elif 'status' in progress:
                        # Optionally print status updates
                        pbar.set_postfix_str(progress['status'])
            logger.info(f'Successfully pulled model: {model_name}')
            output = True
            self.models = self.list_models()  # Update the model list after pulling the model
        except Exception as e:
            logger.error(f'Failed to pull model {model_name}: {e}')
        return output

    @staticmethod
    def create_messages(system_prompt: str, user_prompt: str):
        system_message = {'role': 'system', 'content': system_prompt}
        user_message = {'role': 'user', 'content': user_prompt}
        message_list = [system_message, user_message]
        return message_list

    def send_messages(self,
                      messages: list,
                      model: str,
                      response_format,
                      temperature: float = 0.7,
                      client = None):
        if client is None:
            client = self.create_client()
        try:
            response = client.chat(model=model,
                                   messages=messages,
                                   format=response_format.model_json_schema(),
                                   options={'temperature': temperature})
            message = response2message(response)
            output = response_format.model_validate_json(message)
            output = output.model_dump()
        except Exception as e:
            logger.error(f'Error sending messages: {e}')
            output = None
        else:
            response_dump = response.model_dump()
            output.update({'done': response_dump.get('done', None),
                           'done_reason': response_dump.get('done_reason', None)})
        return output