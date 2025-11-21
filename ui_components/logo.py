import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms


def Logo():
    with antd.Typography.Title(level=1,
                               elem_style=dict(fontSize=24,
                                               padding=8,
                                               margin=0)):
        with antd.Flex(align="center", gap="small", justify="center"):
            ms.Span("🤖")
            ms.Span("RequireGPT")
