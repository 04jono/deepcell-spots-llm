
from dotenv import load_dotenv

load_dotenv()

import os

from autogen import OpenAIWrapper
from autogen.coding import CodeBlock
from autogen.coding.jupyter import (
    DockerJupyterServer,
    JupyterCodeExecutor,
    LocalJupyterServer,
)
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)

from agent_messages import prepare_exploration_message, prepare_modification_message, code_verifier_system_message, code_writer_system_message

max_round = 25  # Maximum number of rounds for the conversation, defined in GroupChat - default is 10

server = LocalJupyterServer()
executor = JupyterCodeExecutor(server, output_dir="autogen-output")


last_summary = ""

def prepare_agents():
    ''' Prepare 3 agents '''
    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": executor
        }, 
        human_input_mode="NEVER",  # Always take human input for this agent for safety.
        # is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    )
    code_writer_agent = ConversableAgent(
        "code_writer",
        system_message=code_writer_system_message,
        llm_config={
            "config_list": [
                {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
    )
    code_verifier_agent = ConversableAgent(
        "code_verifier",
        system_message=code_verifier_system_message,
        llm_config={
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for
        human_input_mode="NEVER",
    )
    
    return code_executor_agent, code_writer_agent, code_verifier_agent

def state_transition(last_speaker, groupchat):
    ''' Transition between speakers in an agent groupchat '''
    messages = groupchat.messages

    if len(messages) <= 1:
        return code_writer_agent

    if last_speaker is code_writer_agent:
        return code_verifier_agent
    elif last_speaker is code_verifier_agent:
        return code_executor_agent
    elif last_speaker is code_executor_agent:
        if "exitcode: 1" in messages[-1]["content"]:
            return code_writer_agent
        else:
            return code_writer_agent
            
for i in range(max_round):
    
    # Phase 1: New function
    
    code_executor_agent, code_writer_agent, code_verifier_agent = prepare_agents()

    group_chat = GroupChat(
        agents=[
            code_executor_agent,
            code_writer_agent,
            code_verifier_agent,
        ],
        messages=[],
        max_round=max_round,
        send_introductions=True,
        speaker_selection_method=state_transition,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        is_termination_msg=lambda msg: (
            "TERMINATE" in msg["content"] if msg["content"] else False
        ),
    )

    chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prepare_exploration_message(context=last_summary), summary_method="reflection_with_llm", summary_args={"summary_prompt": "Summarize the results of this iteration with loss values, and print the code written for the preprocessing function."})
    
    last_summary = chat_result.summary
    
    # Phase 2: Modify existing function
    
    code_executor_agent, code_writer_agent, code_verifier_agent = prepare_agents()

    group_chat = GroupChat(
        agents=[
            code_executor_agent,
            code_writer_agent,
            code_verifier_agent,
        ],
        messages=[],
        max_round=max_round,
        send_introductions=True,
        speaker_selection_method=state_transition,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        is_termination_msg=lambda msg: (
            "TERMINATE" in msg["content"] if msg["content"] else False
        ),
    )

    chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prepare_modification_message(context=last_summary), summary_method="reflection_with_llm", summary_args={"summary_prompt": "Summarize the results of this iteration with loss values, and print the code written for the preprocessing function."})
    
    last_summary = chat_result.summary