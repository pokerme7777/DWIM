import yaml  # type: ignore
from loguru import logger


class RetrieveInContextExamplesByQtypePrompter:
    def __init__(
        self,
        template: str,
        in_context_examples: dict[str, list[dict[str, str]]],
        dataset_records: list[dict],
    ):
        self.template = template
        self.in_context_examples = in_context_examples
        logger.info(
            "Have in-context examples for {} question types", len(in_context_examples)
        )
        self.dataset_records = dataset_records
        logger.info(
            "Building map from question to question type from {} records",
            len(dataset_records),
        )
        self.map_question_to_question_type = {
            record["question"]: record["question_type"] for record in dataset_records
        }

        self._log_in_context_examples_present()

    def _log_in_context_examples_present(self):
        question_types = set(self.map_question_to_question_type.values())
        question_types_with_in_context_examples = set(self.in_context_examples.keys())
        question_types_without_in_context_examples = question_types.difference(
            question_types_with_in_context_examples
        )
        logger.info(
            "Have in-context examples for {} question types, "
            "missing in-context examples for {} question types",
            len(question_types_with_in_context_examples),
            len(question_types_without_in_context_examples),
        )

        if len(question_types_without_in_context_examples):
            logger.info(
                "default in-context examples will be used for the following question types: {}",
                question_types_without_in_context_examples,
            )
            if "default" not in self.in_context_examples:
                raise ValueError(
                    "No default in-context examples found in in_context_examples"
                )

    def _get_question_type(self, question: str) -> str:
        return self.map_question_to_question_type[question]

    def _get_in_context_examples(self, question_type: str) -> list[str]:
        try:
            in_context_examples = self.in_context_examples[question_type]
        except KeyError:
            in_context_examples = self.in_context_examples["default"]
        return [_["text"] for _ in in_context_examples]

    def _format_in_context_example(self, in_context_example: str) -> str:
        # Remove any newlines from the start of the in-context example.
        in_context_example = in_context_example.lstrip("\n")
        # Remove any newlines from the end of the in-context example.
        in_context_example = in_context_example.rstrip("\n")
        # Append a single newline to the end of the in-context example.
        in_context_example = in_context_example + "\n"
        return in_context_example

    def _build_prompt(self, question: str, in_context_examples: list[str], caption_observation: str = "", refered_answer: str = "") -> str:
        template = self.template.replace("INSERT_QUERY_HERE", question)
        template = template.replace(
            "INSERT_IN_CONTEXT_EXAMPLES_HERE", "\n".join(in_context_examples)
        )
        template = template.replace(
            "INSERT_CAPTION_HERE", caption_observation)
        
        # can be ignore if not INSERT_REFER_ANSWER_HERE
        template = template.replace(
            "INSERT_REFER_ANSWER_HERE", refered_answer)
        return template

    ## original version
    # def __call__(self, question: str) -> str: 
    #     question_type = self._get_question_type(question)
    #     in_context_examples = self._get_in_context_examples(question_type)
    #     in_context_examples = [
    #         self._format_in_context_example(_) for _ in in_context_examples
    #     ]
    #     return self._build_prompt(
    #         question=question, in_context_examples=in_context_examples
    #     )
    def __call__(self, question: str, caption_observation: str = "", refered_answer: str="") -> str:
        question_type = self._get_question_type(question)
        in_context_examples = self._get_in_context_examples(question_type)
        in_context_examples = [
            self._format_in_context_example(_) for _ in in_context_examples
        ]
        return self._build_prompt(
            question=question, in_context_examples=in_context_examples, caption_observation=caption_observation, refered_answer=refered_answer
        )

    @classmethod
    def build_from_filepaths(
        cls,
        template_path: str,
        in_context_examples_path: str,
        dataset_records: list[dict],
    ):
        logger.info("Building {} from filepaths", cls.__name__)
        logger.info("template_path: {}", template_path)
        logger.info("in_context_examples_path: {}", in_context_examples_path)
        with open(template_path, "r") as f:
            template = f.read()

        with open(in_context_examples_path, "r") as f:
            in_context_examples = yaml.safe_load(f)

        return cls(
            template=template,
            in_context_examples=in_context_examples,
            dataset_records=dataset_records,
        )


class InsertQueryHerePrompter:
    def __init__(self, template_path):
        try:
            with open(template_path, "r") as f:
                self.template = f.read()
        except FileNotFoundError:
            import os

            raise FileNotFoundError(
                f"Template file not found at path: {template_path}, working directory: {os.getcwd()}"
            )

    def __call__(self, question: str, *args, **kwargs) -> str:
        return self.template.replace("INSERT_QUERY_HERE", question)

    @classmethod
    def build_from_config(cls, config):
        return cls(**config["prompter"])


class IdentityPrompter:
    """
    A prompter that passes the input through unchanged.

    This is useful when you're constructing the prompt outside
    of the program generator.
    """

    def __call__(self, question: str) -> str:
        return question

    @classmethod
    def build_from_config(cls, config):
        return cls()
