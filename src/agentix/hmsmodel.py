""" Retrieve an Okta OAuth 2.0 access token with the password grant """

import os
import json
import logging
import subprocess
from urllib.parse import urlencode
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

OKTA_URL = 'https://login.hms.harvard.edu'
SCOPE = 'openid offline_access'
OKTA_CLIENT_ID = '0oa139tiylzbW6XnX698'
OKTA_AUTH_SERVER_ID = 'aus155lzzptyDTgN3698'
TOKEN_URL = f'{OKTA_URL}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/token'

# Default HMS AI endpoint and model. These can be overridden in the HMSModel constructor.
HMS_API_ENDPOINT = os.environ['HMS_API_ENDPOINT']
HMS_AI_URL = f'{HMS_API_ENDPOINT}/v1'

def get_access_token(username, password, timeout=30):
    """
    Exchanges an HMS username and password for an Okta access token.
    Returns:
        The access token as a string, or None if the credentials were rejected
        or the token could not be retrieved.
    """
    data = urlencode({'client_id': OKTA_CLIENT_ID,
                      'grant_type': 'password',
                      'username': username,
                      'password': password,
                      'scope': SCOPE})
    cmd = ['curl', '--silent', '--show-error', '--request', 'POST',
           '--url', TOKEN_URL,
           '--header', 'Accept: application/json',
           '--header', 'Content-Type: application/x-www-form-urlencoded',
           '--data', '@-']
    token = None
    try:
        result = subprocess.run(cmd,
                                input=data,
                                capture_output=True,
                                text=True,
                                timeout=timeout,
                                check=True)
    except FileNotFoundError:
        logger.error('curl was not found on this system.')
    except subprocess.TimeoutExpired:
        logger.error(f'The request to {TOKEN_URL} timed out after {timeout} seconds.')
    except subprocess.CalledProcessError as e:
        logger.error(f'curl failed with exit code {e.returncode}: {e.stderr.strip()}')
    else:
        # Okta returns HTTP 200 with an error body on bad credentials, so check
        # for the token itself rather than curl's exit status.
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.error(f'Could not parse the Okta response: {result.stdout}')
        else:
            token = response.get('access_token')
            if token is None:
                logger.error(f'Failed to authenticate, check below error\n{result.stdout}')
    return token


class HMSModel:
    """
    Class for managing interactions with the HMS AI endpoint, including listing
    the available models and sending messages to them.
    """
    def __init__(self,
                 token: str,
                 model: str = None,
                 base_url: str = HMS_AI_URL,
                 timeout: int = 30):
        self.token = token
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        if self.model is None:
            self.model = self.list_models()[0]

    def create_client(self):
        client = None
        try:
            client = OpenAI(base_url=self.base_url,
                            api_key=self.token,
                            timeout=self.timeout)
        except openai.OpenAIError as e:
            logger.error(f'Error creating client: {e}')
        return client

    def get_models(self, client=None):
        """
        Retrieves the raw model list from the HMS AI endpoint.
        """
        if client is None:
            client = self.create_client()
        models = None
        try:
            # The client returns a typed page; model_dump gives back the raw JSON.
            models = client.models.list().model_dump()
        except (openai.OpenAIError, AttributeError) as e:
            logger.error(f'Error retrieving model list: {e}')
        return models

    def list_models(self, client=None):
        """
        Lists the names of the models available on the HMS AI endpoint.
        """
        models = self.get_models(client=client)
        model_list = None
        if models is not None:
            model_list = [model.get('id', None) for model in models.get('data', [])]
        return model_list

    @staticmethod
    def create_messages(user_prompt: str, system_prompt: str = None):
        message_list = []
        if system_prompt is not None:
            message_list.append({'role': 'system', 'content': system_prompt})
        message_list.append({'role': 'user', 'content': user_prompt})
        return message_list

    def chat_completion(self,
                        messages: list,
                        model: str = None,
                        temperature: float = 0.7,
                        client = None):
        if client is None:
            client = self.create_client()
        if model is None:
            model = self.model
        output = None
        try:
            response = client.chat.completions.create(model=model,
                                                      messages=messages,
                                                      temperature=temperature)
        except (openai.OpenAIError, AttributeError) as e:
            logger.error(f'Error sending messages: {e}')
        else:
            if response.choices:
                output = response.choices[0].message.content
            else:
                logger.error(f'No message in the response from {model}: {response.model_dump()}')
        return output
