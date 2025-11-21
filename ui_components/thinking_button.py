import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms
import gradio as gr


def ThinkingButton():
    state = gr.State({"enable_thinking": True})
    with antd.Button("Thinking",
                     shape="round",
                     color="primary",
                     variant="solid") as thinking_btn:
        with ms.Slot("icon"):
            antd.Icon("SunOutlined")

    def toggle_thinking(state_value):
        state_value["enable_thinking"] = not state_value["enable_thinking"]
        return gr.update(value=state_value)

    def apply_state_change(state_value):
        return gr.update(
            variant="solid" if state_value["enable_thinking"] else "")

    state.change(fn=apply_state_change, inputs=[state], outputs=[thinking_btn])

    thinking_btn.click(fn=toggle_thinking, inputs=[state], outputs=[state])

    return state
