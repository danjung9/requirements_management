import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms

from config import DEFAULT_SETTINGS, MODEL_OPTIONS, MAX_THINKING_BUDGET, MIN_THINKING_BUDGET, get_text


def SettingsHeader():
    state = gr.State({"open": True})
    with antdx.Sender.Header(title=get_text("Settings", "设置"),
                             open=True) as settings_header:
        with antd.Form(value=DEFAULT_SETTINGS) as settings_form:
            with antd.Form.Item(form_name="model",
                                label=get_text("Chat Model", "对话模型")):
                with antd.Select(options=MODEL_OPTIONS):
                    with ms.Slot("labelRender",
                                 params_mapping="""(option) => ({
                                label: option.label, 
                                link: { href: window.MODEL_OPTIONS_MAP[option.value].link },  
                            })"""):
                        antd.Typography.Text(as_item="label")
                        antd.Typography.Link(get_text("Model Link", "模型链接"),
                                             href_target="_blank",
                                             as_item="link")

            with antd.Form.Item(form_name="thinking_budget",
                                label=get_text("Thinking Budget", "思考预算"),
                                elem_classes="setting-form-thinking-budget"):
                antd.Slider(elem_style=dict(flex=1, marginRight=14),
                            min=MIN_THINKING_BUDGET,
                            max=MAX_THINKING_BUDGET,
                            tooltip=dict(formatter="(v) => `${v}k`"))
                antd.InputNumber(max=MAX_THINKING_BUDGET,
                                 min=MIN_THINKING_BUDGET,
                                 elem_style=dict(width=100),
                                 addon_after="k")
            # with antd.Form.Item(form_name="sys_prompt",
            #                     label=get_text("System Prompt", "系统提示")):
            #     antd.Input.Textarea(auto_size=dict(minRows=3, maxRows=6))

    def close_header(state_value):
        state_value["open"] = False
        return gr.update(value=state_value)

    state.change(fn=lambda state_value: gr.update(open=state_value["open"]),
                 inputs=[state],
                 outputs=[settings_header])

    settings_header.open_change(fn=close_header,
                                inputs=[state],
                                outputs=[state])

    return state, settings_form
