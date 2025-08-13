import json
import os
import time
from typing import Any

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError

model_names_dict = {
        "microsoft/phi-4": "microsoft-phi-4",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "llama-4-mvk-17b-128e-fp8",
        "deepseek-ai/DeepSeek-V3": "deepseek-v3-h200",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3-1-8b-instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama-4-scout-17b-16e-instruct",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-3-1-24b-2503",
        "ibm-granite/granite-guardian-3.2-3b-a800m": "granite-guardian-3-2-3b-a800m",
        "ibm-granite/granite-guardian-3.2-5b": "granite-guardian-3-2-5b-ris",
        "granite-guardian-3-2-5b-ris": "granite-guardian-3-3-8b",
        "openai/gpt-oss-20b": "gpt-oss-20b",
        "openai/gpt-oss-120b": "gpt-oss-120b",
    }

def get_client_model_name(model_name: str) -> str:
    if model_name.startswith("byom-"):
        # Remove "byom-xyz/" initial part of model name, since that's part of the endpoint.
        return "/".join(
            model_name.split("/")[1:]
        )  # This is wrong. since in next iteration
    return model_name

def get_model_name_for_endpoint(model_name: str):
    if model_name in model_names_dict:
        return model_names_dict[model_name]
    if model_name.startswith("byom-"):
        model_name_for_endpoint = model_name.split("/")[0]
        # For RITS BYOM the model name has the following convention:
        # <byom endpoint>/<actual model name>. e.g.
        # byom-gb-iqk-lora/ibm-granite/granite-3.1-8b-instruct
        # at this case we should use https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/byom-gb-iqk-lora/v1/chat/completions
        return model_name_for_endpoint
    return (
        model_name.split("/")[-1]
        .lower()
        .replace("v0.1", "v01")
        .replace("vision-", "")
        .replace(".", "-")
    )

def get_base_url_from_model_name(model_name: str) -> str:
    base_url_template = (
        "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}"
    )
    return base_url_template.format(get_model_name_for_endpoint(model_name))





class RITSCompletionsHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        super().__init__(model_name, temperature)
        self.rits_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_style = ModelStyle.RIS_Completions
        api_key = "4a642e57eb1270a64e62a01571abae8c"
        base_url = get_base_url_from_model_name(self.rits_model_name) + "/v1"
        headers = {"RITS_API_KEY": api_key}

        self.client =  OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )

    def decode_ast(self, result, language="Python"):
        if "FC" in self.rits_model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.rits_model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self.rits_model_name.replace("-FC", ""),
            "temperature": self.temperature,
            "store": False,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        # Original assistant message object (contains `reasoning_content` on DeepSeek).
        message = api_response.choices[0].message

        # Preserve tool_call information but strip the unsupported `reasoning_content` field before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        elif hasattr(message, "reasoning_content"):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        # Capture the reasoning trace so it can be logged to the local result file.
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.rits_model_name,
            temperature=self.temperature,
            store=False,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data

    def _add_reasoning_content_if_available_prompting(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
            # Reasoning content should not be included in the chat history
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": str(response_data["model_responses"]),
            }
