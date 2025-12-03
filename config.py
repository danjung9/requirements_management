import os
from modelscope_studio.components.pro.chatbot import ChatbotActionConfig, ChatbotBotConfig, ChatbotUserConfig, ChatbotWelcomeConfig

from dotenv import load_dotenv

load_dotenv(".env")
# Env
api_key = os.getenv('OPENROUTER_API_KEY')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
QWEN_LOGO_PATH = os.path.join(ASSETS_DIR, "requirementsassistant.png")


# Save history in browser
save_history = True


# Chatbot Config
def user_config(disabled_actions=None):
    return ChatbotUserConfig(
        class_names=dict(content="user-message-content"),
        actions=[
            "copy", "edit",
            ChatbotActionConfig(
                action="delete",
                popconfirm=dict(title="Delete the message",
                                description="Are you sure to delete this message?",
                                okButtonProps=dict(danger=True)))
        ],
        disabled_actions=disabled_actions)


def bot_config(disabled_actions=None):
    return ChatbotBotConfig(actions=[
        "copy", "edit",
        ChatbotActionConfig(
            action="retry",
            popconfirm=dict(
                title="Regenerate the message",
                description="Regenerate the message will also delete all subsequent messages.",
                okButtonProps=dict(danger=True))),
        ChatbotActionConfig(action="delete",
                            popconfirm=dict(
                                title="Delete the message",
                                description="Are you sure to delete this message?",
                                okButtonProps=dict(danger=True)))
    ],
                            avatar=QWEN_LOGO_PATH,
                            disabled_actions=disabled_actions)


def welcome_config():
    return ChatbotWelcomeConfig(
        variant="borderless",
        icon=QWEN_LOGO_PATH,
        title="Hello, I'm Requirements Assistant",
        description="Upload your requirements document and ask a question. I will help show compliance information.",
        prompts=dict(
            title="How can I help you today?",
            styles={
                "list": {
                    "width": '100%',
                },
                "item": {
                    "flex": 1,
                },
            },
            items=[{
                "label":
                "Check Requirements",
                "children": [{
                    "description": "What are lighting requirements when using intermediate or wet-weather tyres?",
                }, {
                    "description": "When using intermediate or wet-weather tyres in a race without a safety car, what are the regulations for the lights?",
                }, {
                    "description": "When there is a safety car during a race, when should lapped cars unlap themselves?",
                }]
            }]),
    )


DEFAULT_SUGGESTIONS = [{
    "label": 'Make a plan',
    "value": 'Make a plan',
    "children": [{
        "label": "Start a business",
        "value": "Help me with a plan to start a business"
    }, {
        "label": "Achieve my goals",
        "value": "Help me with a plan to achieve my goals"
    }, {
        "label": "Successful interview",
        "value": "Help me with a plan for a successful interview"
    }]
}, {
    "label": 'Help me write',
    "value": "Help me write",
    "children": [{
        "label": "Story with a twist ending",
        "value": "Help me write a story with a twist ending"
    }, {
        "label": "Blog post on mental health",
        "value": "Help me write a blog post on mental health"
    }, {
        "label": "Letter to my future self",
        "value": "Help me write a letter to my future self"
    }]
}]

DEFAULT_SYS_PROMPT = "You are a helpful and harmless assistant."

MIN_THINKING_BUDGET = 1

MAX_THINKING_BUDGET = 38

DEFAULT_THINKING_BUDGET = 38

DEFAULT_LOCALE = 'en_US'

DEFAULT_THEME = {
    "token": {
        "colorPrimary": "#6A57FF",
    }
}

DEFAULT_SETTINGS = {
    "sys_prompt": DEFAULT_SYS_PROMPT,
    "uploaded_file": None,
}
