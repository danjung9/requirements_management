from __future__ import annotations
import uuid
import time
import os
import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from config import DEFAULT_LOCALE, DEFAULT_SETTINGS, DEFAULT_THEME, DEFAULT_SUGGESTIONS, save_history, user_config, bot_config, welcome_config, api_key
from ui_components.logo import Logo
from ui_components.settings_header import SettingsHeader
from ui_components.thinking_button import ThinkingButton
from pipelines.requirements_pipe import (
    RAGModel as RequirementsRAGModel,
    Router as RequirementsRouter,
    RequirementsPipeline,
    JiraAgent,
    ComplianceMatrixAgent,
)
from pypdf import PdfReader

## RAG dependencies
import chromadb 
from sentence_transformers import SentenceTransformer

# Global RAG variables (defined before Gradio_Events)
RAG_COLLECTION = None
RAG_EMBEDDER = None
RAG_N_RESULTS = 3 
RAG_MODEL_ID = "zacCMU/miniLM2-ENG3"
RAG_COLLECTION = None
RAG_EMBEDDER = None
client = None
REQUIREMENTS_PIPELINE = None


def load_env_file(env_path: str | None = None):
    """
    Lightweight .env loader to populate os.environ if keys are missing.
    Falls back to the .env that lives next to this file so launching from
    another working directory still picks up keys.
    """
    candidate_paths = []
    if env_path:
        candidate_paths.append(env_path)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths.append(os.path.join(base_dir, ".env"))
        candidate_paths.append(".env")

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key and key not in os.environ:
                        os.environ[key] = value
            print(f"Loaded environment variables from {path}")
            return
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")

# Load .env early so API keys (e.g., OPENROUTER_API_KEY) are available.
load_env_file()
# Basic sanity check so missing keys are obvious in logs.
if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
    print("Warning: OPENROUTER_API_KEY / OPENAI_API_KEY not set; OpenRouter calls will fail.")

MAX_CONTEXT_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_CONTEXT_FILE_CHARACTERS = 6000
SUPPORTED_CONTEXT_FILE_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf"}


def _extract_uploaded_file_path(file_reference):
    if not file_reference:
        return None
    if isinstance(file_reference, list):
        if not file_reference:
            return None
        return _extract_uploaded_file_path(file_reference[0])
    if isinstance(file_reference, str):
        return file_reference
    if isinstance(file_reference, dict):
        return file_reference.get("name") or file_reference.get("path")
    if hasattr(file_reference, "name"):
        return getattr(file_reference, "name")
    return None


def load_context_file(file_reference):
    file_path = _extract_uploaded_file_path(file_reference)
    if not file_path or not os.path.exists(file_path):
        raise gr.Error("Unable to read the uploaded file.")

    file_size = os.path.getsize(file_path)
    if file_size > MAX_CONTEXT_FILE_SIZE:
        raise gr.Error(
            "File too large. Limit is 2 MB.")

    _, ext = os.path.splitext(file_path)
    if ext and ext.lower() not in SUPPORTED_CONTEXT_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_CONTEXT_FILE_EXTENSIONS))
        raise gr.Error(
            f"Unsupported file type. Allowed: {allowed}")

    content = ""
    if ext.lower() == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            content = "\n".join(text_parts)
        except Exception as exc:
            raise gr.Error(f"Unable to read PDF: {exc}")
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    truncated = len(content) > MAX_CONTEXT_FILE_CHARACTERS
    content = content[:MAX_CONTEXT_FILE_CHARACTERS].strip()
    # when uploaded add it to chromadb to! 
    add_documents_to_collection(collection=RAG_COLLECTION, docs=content)

    return {
        "name": os.path.basename(file_path),
        "size": file_size,
        "content": content,
        "truncated": truncated
    }


def resolve_uploaded_file(uploaded_file_value, state_value):
    conversation_id = state_value.get("conversation_id")
    previous_settings = {}
    if conversation_id:
        previous_settings = state_value["conversation_contexts"].get(
            conversation_id, {}).get("settings", {})
    if uploaded_file_value:
        return load_context_file(uploaded_file_value)
    return previous_settings.get("uploaded_file")


def format_file_status(uploaded_file):
    if not uploaded_file:
        return "No file uploaded"
    size_kb = uploaded_file.get("size", 0) / 1024
    size_suffix = f" (~{size_kb:.1f} KB)" if size_kb else ""
    status = f"Using file: {uploaded_file.get('name', 'file')}{size_suffix}"
    if uploaded_file.get("truncated"):
        status += " (content truncated)"
    return status


def format_history(history, sys_prompt, uploaded_file=None):
    messages = []
    system_sections = []
    if sys_prompt:
        system_sections.append(sys_prompt)
    if uploaded_file and uploaded_file.get("content"):
        file_section = (
            f"Reference file ({uploaded_file.get('name', 'file')}):\n"
            f"{uploaded_file.get('content', '')}")
        if uploaded_file.get("truncated"):
            file_section += (
                "\n\n[File content truncated to the first "
                f"{MAX_CONTEXT_FILE_CHARACTERS} characters.]")
        system_sections.append(file_section)
    if system_sections:
        messages.append({
            "role": "system",
            "content": "\n\n".join(system_sections)
        })
    for item in history:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["content"]})
        elif item["role"] == "assistant":
            contents = [{
                "type": "text",
                "text": content["content"]
            } for content in item["content"] if content["type"] == "text"]
            messages.append({
                "role":
                "assistant",
                "content":
                contents[0]["text"] if len(contents) > 0 else ""
            })
    return messages


class Gradio_Events:

    @staticmethod
    def submit(state_value):

        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        settings = state_value["conversation_contexts"][
            state_value["conversation_id"]]["settings"]
        enable_thinking = state_value["conversation_contexts"][
            state_value["conversation_id"]]["enable_thinking"]
        model = settings.get("model")
        messages = format_history(history,
                                  sys_prompt=settings.get("sys_prompt", ""),
                                  uploaded_file=settings.get("uploaded_file"))

        history.append({
            "role":
            "assistant",
            "content": [],
            "key":
            str(uuid.uuid4()),
            "header":
            "Response",
            "loading":
            True,
            "status":
            "pending"
        })

        yield {
            chatbot: gr.update(value=history),
            state: gr.update(value=state_value),
        }
        try:
            pipeline = ensure_pipeline_initialized()

            response = pipeline.stream(messages=messages)
            start_time = time.time()
            reasoning_content = ""
            answer_content = ""
            is_thinking = False
            is_answering = False
            contents = [None, None]
            for chunk in response:
                delta = chunk.output.choices[0].message
                delta_content = (getattr(delta, "content", None)
                                 if not isinstance(delta, dict) else delta.get("content"))
                delta_reason = (getattr(delta, "reasoning_content", None)
                                if not isinstance(delta, dict) else delta.get("reasoning_content"))

                if (not delta_content) and (not delta_reason):
                    pass
                else:
                    if delta_reason:
                        if not is_thinking:
                            contents[0] = {
                                "type": "tool",
                                "content": "",
                                "options": {
                                    "title": "Thinking...",
                                    "status": "pending"
                                },
                                "copyable": False,
                                "editable": False
                            }
                            is_thinking = True
                        reasoning_content += delta_reason
                    if delta_content:
                        if not is_answering:
                            thought_cost_time = "{:.2f}".format(time.time() -
                                                                start_time)
                            if contents[0]:
                                contents[0]["options"]["title"] = f"End of Thought ({thought_cost_time}s)"
                                contents[0]["options"]["status"] = "done"
                            contents[1] = {
                                "type": "text",
                                "content": "",
                            }

                            is_answering = True
                        answer_content += delta_content

                    if contents[0]:
                        contents[0]["content"] = reasoning_content
                    if contents[1]:
                        contents[1]["content"] = answer_content
                history[-1]["content"] = [
                    content for content in contents if content
                ]

                history[-1]["loading"] = False
                yield {
                    chatbot: gr.update(value=history),
                    state: gr.update(value=state_value)
                }
            print("model: ", model, "-", "reasoning_content: ",
                  reasoning_content, "\n", "content: ", answer_content)
            history[-1]["status"] = "done"
            cost_time = "{:.2f}".format(time.time() - start_time)
            history[-1]["footer"] = f"{cost_time}s"
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value),
            }
        except Exception as e:
            print("model: ", model, "-", "Error: ", e)
            history[-1]["loading"] = False
            history[-1]["status"] = "done"
            history[-1]["content"] += [{
                "type":
                "text",
                "content":
                f'<span style="color: var(--color-red-500)">{str(e)}</span>'
            }]
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value)
            }
            return

    @staticmethod
    def add_message(input_value, settings_form_value, thinking_btn_state_value,
                    uploaded_file_value, state_value):
        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            history = []
            state_value["conversation_id"] = random_id
            state_value["conversation_contexts"][
                state_value["conversation_id"]] = {
                    "history": history
                }
            state_value["conversations"].append({
                "label": input_value,
                "key": random_id
            })

        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]

        uploaded_file = resolve_uploaded_file(uploaded_file_value,
                                              state_value)

        state_value["conversation_contexts"][
            state_value["conversation_id"]] = {
                "history": history,
                "settings": {
                    **settings_form_value,
                    "uploaded_file": uploaded_file
                },
                "enable_thinking": thinking_btn_state_value["enable_thinking"]
            }
        history.append({
            "role": "user",
            "content": input_value,
            "key": str(uuid.uuid4())
        })
        yield Gradio_Events.preprocess_submit(clear_input=True)(state_value)

        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def preprocess_submit(clear_input=True):

        def preprocess_submit_handler(state_value):
            history = state_value["conversation_contexts"][
                state_value["conversation_id"]]["history"]
            return {
                **({
                    input:
                    gr.update(value=None, loading=True) if clear_input else gr.update(loading=True),
                } if clear_input else {}),
                conversations:
                gr.update(active_key=state_value["conversation_id"],
                          items=list(
                              map(
                                  lambda item: {
                                      **item,
                                      "disabled":
                                      True if item["key"] != state_value[
                                          "conversation_id"] else False,
                                  }, state_value["conversations"]))),
                add_conversation_btn:
                gr.update(disabled=True),
                clear_btn:
                gr.update(disabled=True),
                conversation_delete_menu_item:
                gr.update(disabled=True),
                chatbot:
                gr.update(value=history,
                          bot_config=bot_config(
                              disabled_actions=['edit', 'retry', 'delete']),
                          user_config=user_config(
                              disabled_actions=['edit', 'delete'])),
                state:
                gr.update(value=state_value),
            }

        return preprocess_submit_handler

    @staticmethod
    def postprocess_submit(state_value):
        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        return {
            input:
            gr.update(loading=False),
            conversation_delete_menu_item:
            gr.update(disabled=False),
            clear_btn:
            gr.update(disabled=False),
            conversations:
            gr.update(items=state_value["conversations"]),
            add_conversation_btn:
            gr.update(disabled=False),
            chatbot:
            gr.update(value=history,
                      bot_config=bot_config(),
                      user_config=user_config()),
            state:
            gr.update(value=state_value),
        }

    @staticmethod
    def cancel(state_value):
        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        history[-1]["loading"] = False
        history[-1]["status"] = "done"
        history[-1]["footer"] = "Chat completion paused"
        return Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def delete_message(state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        history = history[:index] + history[index + 1:]

        state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"] = history

        return gr.update(value=state_value)

    @staticmethod
    def edit_message(state_value, chatbot_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        history[index]["content"] = chatbot_value[index]["content"]
        return gr.update(value=state_value)

    @staticmethod
    def regenerate_message(settings_form_value, thinking_btn_state_value,
                           uploaded_file_value, state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        history = history[:index]

        uploaded_file = resolve_uploaded_file(uploaded_file_value,
                                              state_value)

        state_value["conversation_contexts"][
            state_value["conversation_id"]] = {
                "history": history,
                "settings": {
                    **settings_form_value,
                    "uploaded_file": uploaded_file
                },
                "enable_thinking": thinking_btn_state_value["enable_thinking"]
            }

        yield Gradio_Events.preprocess_submit()(state_value)
        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def select_suggestion(input_value, e: gr.EventData):
        input_value = input_value[:-1] + e._data["payload"][0]
        return gr.update(value=input_value)

    @staticmethod
    def apply_prompt(e: gr.EventData):
        return gr.update(value=e._data["payload"][0]["value"]["description"])

    @staticmethod
    def new_chat(thinking_btn_state, state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_id"] = ""
        thinking_btn_state["enable_thinking"] = True
        return (
            gr.update(active_key=state_value["conversation_id"]),
            gr.update(value=None),
            gr.update(value={**DEFAULT_SETTINGS}),
            gr.update(value=None),
            gr.update(value=format_file_status(None)),
            gr.update(value=thinking_btn_state),
            gr.update(value=state_value),
        )

    @staticmethod
    def select_conversation(thinking_btn_state_value, state_value,
                            e: gr.EventData):
        active_key = e._data["payload"][0]
        if state_value["conversation_id"] == active_key or (
                active_key not in state_value["conversation_contexts"]):
            return gr.skip()
        state_value["conversation_id"] = active_key
        conversation = state_value["conversation_contexts"][active_key]
        thinking_btn_state_value["enable_thinking"] = conversation[
            "enable_thinking"]
        settings = conversation.get("settings") or {**DEFAULT_SETTINGS}
        return (
            gr.update(active_key=active_key),
            gr.update(value=conversation["history"]),
            gr.update(value=settings),
            gr.update(value=None),
            gr.update(value=format_file_status(settings.get("uploaded_file"))),
            gr.update(value=thinking_btn_state_value),
            gr.update(value=state_value),
        )

    @staticmethod
    def click_conversation_menu(state_value, e: gr.EventData):
        conversation_id = e._data["payload"][0]["key"]
        operation = e._data["payload"][1]["key"]
        if operation == "delete":
            del state_value["conversation_contexts"][conversation_id]

            state_value["conversations"] = [
                item for item in state_value["conversations"]
                if item["key"] != conversation_id
            ]

            if state_value["conversation_id"] == conversation_id:
                state_value["conversation_id"] = ""
                return (
                    gr.update(items=state_value["conversations"],
                              active_key=state_value["conversation_id"]),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=format_file_status(None)),
                    gr.update(value=state_value),
                )
            else:
                return (
                    gr.update(items=state_value["conversations"]),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.update(value=state_value),
                )
        return gr.skip()

    @staticmethod
    def toggle_settings_header(settings_header_state_value):
        settings_header_state_value[
            "open"] = not settings_header_state_value["open"]
        return gr.update(value=settings_header_state_value)

    @staticmethod
    def clear_conversation_history(state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"] = []
        return gr.update(value=None), gr.update(value=state_value)

    @staticmethod
    def update_browser_state(state_value):

        return gr.update(value=dict(
            conversations=state_value["conversations"],
            conversation_contexts=state_value["conversation_contexts"]))

    @staticmethod
    def apply_browser_state(browser_state_value, state_value):
        state_value["conversations"] = browser_state_value["conversations"]
        state_value["conversation_contexts"] = browser_state_value[
            "conversation_contexts"]
        return gr.update(
            items=browser_state_value["conversations"]), gr.update(
                value=state_value)

    @staticmethod
    def preview_uploaded_file(uploaded_file_value):
        if not uploaded_file_value:
            return gr.update(value=format_file_status(None))
        uploaded_file = load_context_file(uploaded_file_value)
        return gr.update(value=format_file_status(uploaded_file))

    @staticmethod
    def remove_uploaded_file(state_value):
        conversation_id = state_value.get("conversation_id")
        if conversation_id and conversation_id in state_value[
                "conversation_contexts"]:
            state_value["conversation_contexts"][conversation_id].setdefault(
                "settings", {**DEFAULT_SETTINGS})
            state_value["conversation_contexts"][conversation_id]["settings"][
                "uploaded_file"] = None
        return gr.update(value=None), gr.update(
            value=format_file_status(None)), gr.update(value=state_value)


css = """
.gradio-container {
  padding: 0 !important;
}

.gradio-container > main.fillable {
  padding: 0 !important;
}

#chatbot {
  height: calc(100vh - 21px - 16px);
  max-height: 1500px;
}

#chatbot .chatbot-conversations {
  height: 100vh;
  background-color: var(--ms-gr-ant-color-bg-layout);
  padding-left: 4px;
  padding-right: 4px;
}


#chatbot .chatbot-conversations .chatbot-conversations-list {
  padding-left: 0;
  padding-right: 0;
}

#chatbot .chatbot-chat {
  padding: 32px;
  padding-bottom: 0;
  height: 100%;
}

@media (max-width: 768px) {
  #chatbot .chatbot-chat {
      padding: 0;
  }
}

#chatbot .chatbot-chat .chatbot-chat-messages {
  flex: 1;
}


#chatbot .setting-form-thinking-budget .ms-gr-ant-form-item-control-input-content {
    display: flex;
    flex-wrap: wrap;
}

#chatbot .setting-form-file-upload input[type="file"] {
    padding: 4px;
}

#chatbot .setting-form-file-status {
    font-size: 12px;
    color: var(--ms-gr-ant-color-text-tertiary);
    margin-top: 4px;
}
"""

with gr.Blocks(css=css, fill_width=True) as demo:
    state = gr.State({
        "conversation_contexts": {},
        "conversations": [],
        "conversation_id": "",
    })

    with ms.Application(), antdx.XProvider(
            theme=DEFAULT_THEME, locale=DEFAULT_LOCALE), ms.AutoLoading():
        with antd.Row(gutter=[20, 20], wrap=False, elem_id="chatbot"):
            # Left Column
            with antd.Col(md=dict(flex="0 0 260px", span=24, order=0),
                          span=0,
                          elem_style=dict(width=0),
                          order=1):
                with ms.Div(elem_classes="chatbot-conversations"):
                    with antd.Flex(vertical=True,
                                   gap="small",
                                   elem_style=dict(height="100%")):
                        # Logo
                        Logo()

                        # New Conversation Button
                        with antd.Button(value=None,
                                         color="primary",
                                         variant="filled",
                                         block=True) as add_conversation_btn:
                            ms.Text("New Conversation")
                            with ms.Slot("icon"):
                                antd.Icon("PlusOutlined")

                        # Conversations List
                        with antdx.Conversations(
                                elem_classes="chatbot-conversations-list",
                        ) as conversations:
                            with ms.Slot('menu.items'):
                                with antd.Menu.Item(
                                        label="Delete", key="delete",
                                        danger=True
                                ) as conversation_delete_menu_item:
                                    with ms.Slot("icon"):
                                        antd.Icon("DeleteOutlined")
            # Right Column
            with antd.Col(flex=1, elem_style=dict(height="100%")):
                with antd.Flex(vertical=True,
                               gap="small",
                               elem_classes="chatbot-chat"):
                    # Chatbot
                    chatbot = pro.Chatbot(elem_classes="chatbot-chat-messages",
                                          height=0,
                                          welcome_config=welcome_config(),
                                          user_config=user_config(),
                                          bot_config=bot_config())

                    # Input
                    with antdx.Suggestion(
                            items=DEFAULT_SUGGESTIONS,
                            # onKeyDown Handler in Javascript
                            should_trigger="""(e, { onTrigger, onKeyDown }) => {
                      switch(e.key) {
                        case '/':
                          onTrigger()
                          break
                        case 'ArrowRight':
                        case 'ArrowLeft':
                        case 'ArrowUp':
                        case 'ArrowDown':
                          break;
                        default:
                          onTrigger(false)
                      }
                      onKeyDown(e)
                    }""") as suggestion:
                        with ms.Slot("children"):
                            with antdx.Sender(placeholder="Enter \"/\" to get suggestions") as input:
                                with ms.Slot("header"):
                                    settings_header_state, settings_form, context_file, file_status, remove_file_btn = SettingsHeader(
                                    )
                                with ms.Slot("prefix"):
                                    with antd.Flex(
                                            gap=4,
                                            wrap=True,
                                            elem_style=dict(maxWidth='40vw')):
                                        with antd.Button(
                                                value=None,
                                                type="text") as setting_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("SettingOutlined")
                                        with antd.Button(
                                                value=None,
                                                type="text") as clear_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("ClearOutlined")
                                        thinking_btn_state = ThinkingButton()

    # Events Handler
    # Browser State Handler
    if save_history:
        browser_state = gr.BrowserState(
            {
                "conversation_contexts": {},
                "conversations": [],
            },
            storage_key="chat_demo_storage")
        state.change(fn=Gradio_Events.update_browser_state,
                     inputs=[state],
                     outputs=[browser_state])

        demo.load(fn=Gradio_Events.apply_browser_state,
                  inputs=[browser_state, state],
                  outputs=[conversations, state])

    # Conversations Handler
    add_conversation_btn.click(fn=Gradio_Events.new_chat,
                               inputs=[thinking_btn_state, state],
                               outputs=[
                                   conversations, chatbot, settings_form,
                                   context_file, file_status,
                                   thinking_btn_state, state
                               ])
    conversations.active_change(fn=Gradio_Events.select_conversation,
                                inputs=[thinking_btn_state, state],
                                outputs=[
                                    conversations, chatbot, settings_form,
                                    context_file, file_status,
                                    thinking_btn_state, state
                                ])
    conversations.menu_click(fn=Gradio_Events.click_conversation_menu,
                             inputs=[state],
                             outputs=[
                                 conversations, chatbot, context_file,
                                 file_status, state
                             ])
    # Chatbot Handler
    chatbot.welcome_prompt_select(fn=Gradio_Events.apply_prompt,
                                  outputs=[input])

    chatbot.delete(fn=Gradio_Events.delete_message,
                   inputs=[state],
                   outputs=[state])
    chatbot.edit(fn=Gradio_Events.edit_message,
                 inputs=[state, chatbot],
                 outputs=[state])

    regenerating_event = chatbot.retry(
        fn=Gradio_Events.regenerate_message,
        inputs=[settings_form, thinking_btn_state, context_file, state],
        outputs=[
            input, clear_btn, conversation_delete_menu_item,
            add_conversation_btn, conversations, chatbot, state
        ])

    # Input Handler
    submit_event = input.submit(
        fn=Gradio_Events.add_message,
        inputs=[input, settings_form, thinking_btn_state, context_file, state],
        outputs=[
            input, clear_btn, conversation_delete_menu_item,
            add_conversation_btn, conversations, chatbot, state
        ])
    input.cancel(fn=Gradio_Events.cancel,
                 inputs=[state],
                 outputs=[
                     input, conversation_delete_menu_item, clear_btn,
                     conversations, add_conversation_btn, chatbot, state
                 ],
                 cancels=[submit_event, regenerating_event],
                 queue=False)
    # Input Actions Handler
    setting_btn.click(fn=Gradio_Events.toggle_settings_header,
                      inputs=[settings_header_state],
                      outputs=[settings_header_state])
    clear_btn.click(fn=Gradio_Events.clear_conversation_history,
                    inputs=[state],
                    outputs=[chatbot, state])
    context_file.change(fn=Gradio_Events.preview_uploaded_file,
                        inputs=[context_file],
                        outputs=[file_status])
    remove_file_btn.click(fn=Gradio_Events.remove_uploaded_file,
                          inputs=[state],
                          outputs=[context_file, file_status, state])
    suggestion.select(fn=Gradio_Events.select_suggestion,
                      inputs=[input],
                      outputs=[input])


class CustomSBERTEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    A custom wrapper to use a SentenceTransformer model as the embedding function 
    for ChromaDB, satisfying ChromaDB's interface requirements.
    """
    def __init__(self, model: SentenceTransformer):
        self._model = model
    
    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Outputs a list of lists of floats as ChromaDB expects
        embeddings = self._model.encode(texts, convert_to_tensor=False).tolist()
        return embeddings
    
    def name(self) -> str:
        return "custom_sbert_wrapper"


class ChromaRetriever:
    """Thin wrapper to fetch top-n docs from ChromaDB."""

    def __init__(self, collection: chromadb.api.models.Collection | None,
                 n_results: int = RAG_N_RESULTS):
        self.collection = collection
        self.n_results = n_results

    def search(self, query: str) -> list[str]:
        if not self.collection or not query:
            return []
        results = retrieve_documents(self.collection,
                                     query=query,
                                     n_results=self.n_results)
        docs = results.get("documents") or []
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        return docs


class LocalSummarizer:
    """Lightweight summarizer using retrieved context without external calls."""

    def summarize(self, query: str, docs: list[str]) -> str:
        context = "\n\n".join(docs) if docs else "No retrieved context."
        return (
            "Requirements summary (heuristic):\n"
            f"Inquiry: {query}\n"
            f"Context:\n{context}"
        )


def add_documents_to_collection(collection: chromadb.Collection | None, docs: str):
    """
    Chunks a single document string and adds it to the ChromaDB collection.
    """
    if not collection:
        print("RAG Collection is not initialized. Skipping document addition.")
        return
        
    chunks = split_document_into_chunks(docs)
    if not chunks:
        return

    # Create unique IDs for each chunk
    ids = [f"doc_{uuid.uuid4()}" for _ in range(len(chunks))]
    
    try:
        collection.add(
            documents=chunks,
            ids=ids,
            # metadata can be added here, e.g., source file name
        )
        print(f"Added {len(chunks)} chunks to ChromaDB.")
    except Exception as e:
        print(f"Failed to add documents to ChromaDB: {e}")

def retrieve_documents(collection: chromadb.api.models.Collection | None,
                       query: str,
                       n_results: int = 5) -> dict:
    """
    Retrieves the top N relevant documents from the ChromaDB collection based on a query.
    """
    if not collection or not query:
        return {"documents": [], "distances": []}
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'distances']
    )
    return results

def split_document_into_chunks(text: str, chunk_size=300, chunk_overlap=50) -> list[str]:
    """Simple text splitting for RAG chunking."""
    if not text:
        return []
    
    # A simplified chunking logic: split by sentence or paragraph and then group
    # For robust splitting, consider libraries like LangChain's TextSplitters.
    
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


def init_rag_if_needed():
    """Initialize embedder and Chroma collection if not already set."""
    global RAG_EMBEDDER, RAG_COLLECTION, client
    if RAG_COLLECTION is not None and RAG_EMBEDDER is not None:
        return
    try:
        RAG_EMBEDDER = SentenceTransformer(RAG_MODEL_ID)
        custom_ef = CustomSBERTEmbeddingFunction(RAG_EMBEDDER)
        client = chromadb.Client()
        RAG_COLLECTION = client.get_or_create_collection(
            name="engineering_corpus_rag",
            embedding_function=custom_ef)
        print("RAG initialized.")
    except Exception as e:
        print(f"FATAL RAG SETUP ERROR: {e}")
        print("RAG functionality disabled.")
        RAG_COLLECTION = None
        RAG_EMBEDDER = None
        client = None


def ensure_pipeline_initialized():
    """Lazy-init the RAG -> router -> agent pipeline."""
    global REQUIREMENTS_PIPELINE
    if REQUIREMENTS_PIPELINE:
        return REQUIREMENTS_PIPELINE
    load_env_file()
    init_rag_if_needed()
    retriever = ChromaRetriever(RAG_COLLECTION, n_results=RAG_N_RESULTS)
    summarizer = LocalSummarizer()
    router = RequirementsRouter()
    jira_agent = JiraAgent()
    matrix_agent = ComplianceMatrixAgent()
    REQUIREMENTS_PIPELINE = RequirementsPipeline(
        rag_model=RequirementsRAGModel(retriever=retriever, llm=summarizer),
        router=router,
        jira_agent=jira_agent,
        matrix_agent=matrix_agent,
    )
    return REQUIREMENTS_PIPELINE

if __name__ == "__main__":

    ensure_pipeline_initialized()

    demo.queue(default_concurrency_limit=100,
               max_size=100).launch(ssr_mode=False, max_threads=100)
