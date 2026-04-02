import os
from typing import Callable, cast
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
import logging
from openai import OpenAI  # type: ignore

logger = logging.getLogger(__name__)


class ChatGptProgramGenerator:
    def __init__(self, prompter: Callable[[str], str], **kwargs):
        self.prompter = prompter
        # TODO: This is unclean, we should do it by constructing
        # the client using a nested configuration and hydra.
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")
        else:
            timeout = None
        self.kwargs = kwargs
        self.client = OpenAI(timeout=timeout)
        # openai.api_key = os.environ["OPENAI_API_KEY"]

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def generate(self, query: str) -> str:
        prompt = self.prompter(query)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs
        )
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     **self.kwargs
        # )

        synthesized_program = completion.choices[0].message.content
        return cast(str, synthesized_program)


class ChatGptQuestionAnswer:
    def __init__(self):
        self.code_root = os.path.dirname(os.path.dirname(__file__))
        self.prompts_dir = os.path.join(self.code_root, "prompts")
        self.template_path = os.path.join(self.prompts_dir, "gpt3_qa.txt")
        with open(self.template_path, "r") as f:
            self.template = f.read()
        self.client = OpenAI()

    def fill_template(self, question: str) -> str:
        return self.template.format(question)

    def __call__(self, question: str) -> str:
        prompt = self.fill_template(question)

        # completion = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        # )
        # completion = self.client.chat.completions.create(
        #     model="gpt-4o-mini-2024-07-18", messages=[{"role": "user", "content": prompt}]
        # )
        completion = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content  # type: ignore


class ChatGptProcessGuess:
    def __init__(self):
        self.code_root = os.path.dirname(os.path.dirname(__file__))
        self.prompts_dir = os.path.join(self.code_root, "prompts")
        self.template_path = os.path.join(self.prompts_dir, "gpt3_process_guess.txt")
        with open(self.template_path, "r") as f:
            self.template = f.read()
        self.client = OpenAI()

    def fill_template(self, question: str, guesses: list[str]) -> str:
        # This looks weird, but we're just following the ViperGPT
        # implementation here. It allows 2 guesses and then chooses
        # between the two guesses.
        guess_a, guess_b = guesses
        return self.template.format(question, guess_a, guess_b)

    def __call__(self, question: str, guesses: list[str]) -> str:
        prompt = self.fill_template(question, guesses)

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content  # type: ignore


if __name__ == "__main__":
    openai_answer = ChatGptQuestionAnswer()
    print(openai_answer("which part of dog would play with frisbee?"))