from typing import Callable, Optional
from loguru import logger
import torch


# TODO: There need to be guards here (beartype?) to make sure that mistakes in hydra
# config specifications don't cause undecipherable errors down the line. For example
# if you mess up the config, local_llm will be a str and everything will error out
# during inference, but because all of inference is wrapped in a try/except, it'll
# just look like the planner is bad.
class LocalLLMProgramGenerator:
    def __init__(
        self, prompter: Callable[[str], str], local_llm, tokenizer, generate_kwargs=None
    ):
        self.prompter = prompter
        self.local_llm = local_llm
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs or dict()
        self.last_failed_program: Optional[str] = None

        logger.info(
            "Instantiated LocalLLMProgramGenerator with local_llm={}",
            local_llm.__class__.__name__,
        )

    def postprocess_text(self, input_string: str) -> Optional[str]:
        """
        Function to process the string, keeping only things before the double newline.

        Args:
        input_string (str): The string to be processed.

        Returns:
        Optional[str]: The processed string, or None if the string does not contain
        a double newline.
        """
        logger.debug("raw_program={}", input_string)
        split_string = input_string.split("\n\n", 1)
        if len(split_string) > 1:
            # There will be a newline at the start of the program,
            # strip it.
            return split_string[0].lstrip("\n")
        else:
            self.last_failed_program = input_string
        # The program could still be correct, if it takes up the
        # entire context. In this case, it ends with a </s>. Occurs
        # for `twoCommon` questions.
        if len(input_string.split("</s")) > 1:
            return input_string.split("</s")[0]
        # Return `input_string` anyway, because maybe the program
        # is correct...
        return input_string

    @torch.no_grad()
    def generate(self, query: str):
        tokenized = self.tokenizer(self.prompter(query), return_tensors="pt")
        generation_output = self.local_llm.generate(
            input_ids=torch.as_tensor(tokenized.input_ids).to(self.local_llm.device),
            attention_mask=torch.as_tensor(tokenized.attention_mask).to(
                self.local_llm.device
            ),
            **self.generate_kwargs,
        )

        # We have to postprocess the string because LLAMA doesn't know when to stop.
        # Because of the prompt design, we can assume that everything up to
        # the first double newline is the actual program, and eveything after
        # is extraneous.
        new_tokens_only = generation_output[0][
            tokenized.input_ids.shape[1] :
        ]  # Grab only the output tokens.
        new_text_only = self.tokenizer.decode(new_tokens_only)
        program = self.postprocess_text(new_text_only)

        return program
