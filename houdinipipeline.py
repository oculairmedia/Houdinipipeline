from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import hrpyc
import rpyc
import logging
from openai import OpenAI
import time
import requests

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Pipeline:
    class Valves(BaseModel):
        HOUDINI_SERVER: str = "localhost"
        HOUDINI_PORT: int = 18811
        OPENAI_API_KEY: str = ""
        OPENAI_MODEL: str = "gpt-3.5-turbo"

    def __init__(self):
        self.name = "Houdini Scene Creation Pipeline with OpenAI"
        self.before_scene = []
        self.after_scene = []

        # Initialize valves with default values and environment variables
        self.valves = self.Valves(
            HOUDINI_SERVER=os.getenv("HOUDINI_SERVER", "localhost"),
            HOUDINI_PORT=int(os.getenv("HOUDINI_PORT", 18811)),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
        self.connection = None
        self.hou = None
        self.openai_client = None

    async def on_startup(self):
        logging.info(f"Pipeline '{self.name}' starting up.")
        logging.info(f"Houdini connection: {self.valves.HOUDINI_SERVER}:{self.valves.HOUDINI_PORT}")
        logging.info(f"OpenAI Model: {self.valves.OPENAI_MODEL}")
        self.openai_client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
        await self.validate_openai_api_key()

    async def on_shutdown(self):
        logging.info(f"Pipeline '{self.name}' shutting down.")
        if self.connection:
            self.connection.close()
            logging.info("Houdini connection closed.")

    async def validate_openai_api_key(self):
        try:
            # Make a simple API call to validate the key
            self.openai_client.models.list()
            logging.info("OpenAI API key is valid.")
        except Exception as e:
            logging.error(f"Error validating OpenAI API key: {str(e)}")
            raise

    def connect_to_houdini(self):
        try:
            logging.info(f"Attempting to connect to Houdini RPC server at {self.valves.HOUDINI_SERVER}:{self.valves.HOUDINI_PORT}")
            self.connection, self.hou = hrpyc.import_remote_module(self.valves.HOUDINI_SERVER, self.valves.HOUDINI_PORT)
            logging.info("Successfully connected to Houdini RPC server")
            logging.info(f"Houdini version: {self.hou.applicationVersion()}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Houdini RPC server: {str(e)}")
            return False

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logging.info(f"Received user message: {user_message}")

        if body.get("title", False):
            return "Houdini Scene Creation Pipeline with OpenAI"

        try:
            if not self.connect_to_houdini():
                return "Failed to connect to Houdini RPC server. Please check the server status and try again."

            # Ensure a valid model name is used
            model = self.get_valid_model_name(model_id)
            logging.info(f"Using OpenAI model: {model}")

            # Generate Python code from the user's description
            generated_code = self.generate_python_code(user_message, model)
            logging.info(f"Generated code:\n{generated_code}")

            if generated_code.startswith("# Failed to generate code"):
                return generated_code

            # Execute the generated code in Houdini
            execution_result = self.execute_code_in_houdini(generated_code)
            logging.info(f"Execution result: {execution_result}")

            # Retrieve scene changes
            scene_info = self.get_scene_changes()
            logging.info(f"Scene changes:\n{scene_info}")

            # Return the scene information to the user
            return scene_info
        except Exception as e:
            logging.error(f"Error in pipe method: {str(e)}")
            return f"An error occurred: {str(e)}"

    def get_valid_model_name(self, model_id: str) -> str:
        """
        Ensure a valid model name is used.
        """
        if model_id and model_id.strip() and model_id != "houdinipipelinev_combined":
            return model_id.strip()
        return self.valves.OPENAI_MODEL

    def generate_python_code(self, description: str, model: str) -> str:
        """
        Use OpenAI to generate Houdini Python code based on the user's description.

        Example of expected Houdini Python code:
        ```python
        obj = hou.node("/obj")
        geo = obj.createNode("geo", "my_geometry")
        sphere = geo.createNode("sphere", "my_sphere")
        sphere.parm("type").set(1)  # Set to 'Polygon Mesh'
        sphere.parm("radx").set(2.0)
        sphere.parm("rady").set(2.0)
        sphere.parm("radz").set(2.0)
        sphere.setDisplayFlag(True)
        sphere.setRenderFlag(True)
        hou.hipFile.save('e:/PROJECTS/houdini python integration/houdini connection test/sphere_scene.hip')
        ```
        """
        prompt = f"""
        Generate Python code for Houdini to create a scene based on the following description:
        "{description}"
        
        Use the hou module to create and manipulate nodes. Include code to save the scene as a .hip file.
        Ensure the code is valid Houdini Python and follows best practices.
        Provide only the Python code without any additional explanations or markdown formatting.

        Here's an example of the kind of code structure expected:

        obj = hou.node("/obj")
        geo = obj.createNode("geo", "my_geometry")
        sphere = geo.createNode("sphere", "my_sphere")
        sphere.parm("type").set(1)  # Set to 'Polygon Mesh'
        sphere.parm("radx").set(2.0)
        sphere.parm("rady").set(2.0)
        sphere.parm("radz").set(2.0)
        sphere.setDisplayFlag(True)
        sphere.setRenderFlag(True)
        hou.hipFile.save('path/to/save/scene.hip')

        Adapt this structure to create the scene described, using appropriate node types and parameter settings.
        """

        messages = [
            {"role": "system", "content": "You are a Houdini Python expert. Generate code that creates scenes in Houdini based on user descriptions."},
            {"role": "user", "content": prompt}
        ]

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].message.content.strip()
            except requests.exceptions.RequestException as e:
                logging.error(f"Network error when calling OpenAI API (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error("Max retries reached. Failed to generate code.")
                    return "# Failed to generate code due to network issues."
            except Exception as e:
                logging.error(f"Error generating code with OpenAI: {str(e)}")
                return f"# Failed to generate code: {str(e)}"

    def serialize_scene(self):
        """
        Serialize the scene's nodes and parameters.
        """
        def node_to_dict(node):
            return {
                'path': node.path(),
                'type': node.type().name(),
                'name': node.name(),
                'parameters': {parm.name(): parm.eval() for parm in node.parms()},
                'children': [node_to_dict(child) for child in node.children()],
            }

        obj = self.hou.node("/obj")
        scene_dict = [node_to_dict(child) for child in obj.children()]
        return scene_dict

    def execute_code_in_houdini(self, code: str) -> str:
        """
        Execute the generated code in Houdini using hrpyc.
        """
        try:
            # Serialize the scene before execution
            self.before_scene = self.serialize_scene()

            # Execute code
            exec(code, {'hou': self.hou})

            # Serialize the scene after execution
            self.after_scene = self.serialize_scene()

            execution_result = "Code executed successfully in Houdini."
        except Exception as e:
            execution_result = f"An error occurred while executing code in Houdini: {e}"
            logging.error(execution_result)
            logging.debug("Error details:", exc_info=True)

        return execution_result

    def get_scene_changes(self) -> str:
        """
        Retrieve information about the scene changes.
        """
        # Compare the scene before and after code execution
        before_paths = {node['path'] for node in self.before_scene}
        after_paths = {node['path'] for node in self.after_scene}

        new_nodes = after_paths - before_paths
        changes = []

        for node in self.after_scene:
            if node['path'] in new_nodes:
                changes.append(f"New node created: {node['path']} of type {node['type']}")
                # Add some basic parameter information
                for param_name, param_value in node['parameters'].items():
                    changes.append(f"  - {param_name}: {param_value}")

        if changes:
            scene_info = "\n".join(changes)
        else:
            scene_info = "No changes detected."

        return scene_info

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        pipeline = Pipeline()
        await pipeline.on_startup()
        result = pipeline.pipe("Create a red sphere with radius 2 and a blue cube with side length 1, positioned 3 units above the sphere", None, [], {})
        print(result)
        await pipeline.on_shutdown()

    asyncio.run(main())