from tiktoken import Encoding, encoding_for_model


def get_tokens(encoding: Encoding, text: str) -> list[int]:
    return encoding.encode(text)


def decode_tokens(encoding: Encoding, text: str) -> str:
    return encoding.decode(text)


if __name__ == '__main__':
    text = "안녕하세요, 저는 ChatGPT입니다."

    model_00 = "gpt-3.5-turbo"
    model_01 = "gpt-4"
    model_02 = "gpt-4o"
    model_03 = "gpt-4o-mini"

    encoding_00 = encoding_for_model(model_00)
    encoding_01 = encoding_for_model(model_01)
    encoding_02 = encoding_for_model(model_02)
    encoding_03 = encoding_for_model(model_03)

    tokens_00 = get_tokens(encoding_00, text)
    tokens_01 = get_tokens(encoding_01, text)
    tokens_02 = get_tokens(encoding_02, text)
    tokens_03 = get_tokens(encoding_03, text)

    print(f"model: {model_00}, token length: {len(tokens_00)}, tokens: {tokens_00}")
    print(f"model: {model_01}, token length: {len(tokens_01)}, tokens: {tokens_01}")
    print(f"model: {model_02}, token length: {len(tokens_02)}, tokens: {tokens_02}")
    print(f"model: {model_03}, token length: {len(tokens_03)}, tokens: {tokens_03}")
