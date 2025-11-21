import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms

from config import DEFAULT_SETTINGS


def SettingsHeader():
    state = gr.State({"open": True})
    with antdx.Sender.Header(title="Settings",
                             open=True) as settings_header:
        with antd.Form(value=DEFAULT_SETTINGS) as settings_form:
            with antd.Form.Item(label="Knowledge File"):
                with antd.Flex(gap="small", align="center", wrap=True):
                    context_file = gr.File(label=None,
                                           file_count="single",
                                           file_types=[".txt", ".md", ".json", ".csv"],
                                           type="filepath",
                                           elem_classes="setting-form-file-upload")
                    remove_file_btn = antd.Button("Remove",
                                                  type="text",
                                                  danger=True)
                file_status = gr.Markdown("No file uploaded",
                                          elem_classes="setting-form-file-status")

    def close_header(state_value):
        state_value["open"] = False
        return gr.update(value=state_value)

    state.change(fn=lambda state_value: gr.update(open=state_value["open"]),
                 inputs=[state],
                 outputs=[settings_header])

    settings_header.open_change(fn=close_header,
                                inputs=[state],
                                outputs=[state])

    return state, settings_form, context_file, file_status, remove_file_btn
