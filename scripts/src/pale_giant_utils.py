import json
from tqdm.auto import tqdm
import torch
import time
import logging
from loguru import logger
from collections.abc import Sequence
import inspect
import sys
from collections import Counter
from typing import Any, Iterator, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from typing import Callable
import re
import builtins
import ast


class EnumeratorWithRepeats:
    def __init__(self, sequence: Sequence, repeat: int = 1) -> None:
        self.iterable = sequence
        self.repeat = repeat
        self.length = len(sequence) * repeat

    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        idx = 0
        for item in self.iterable:
            for _ in range(self.repeat):
                yield idx, item
                idx += 1

    def __len__(self) -> int:
        return self.length


class ResumableIterator:
    """
    ResumableEnumerator iterates over a list of dictionaries,
    each containing a 'question_id' key to ensure that each question is iterated over
    a fixed number of times, allowing resuming from a previous run where each question
    was iterated over a variable number of times.

    Parameters:
    - sequence: List of dictionaries to iterate over. Each dictionary must have a 'question_id' key.
    - iterations_per_record: Number of times each record should be iterated over.
    - seen_records: List of dictionaries representing records that have been seen in previous runs.

    Methods:
    - __iter__: Yields each item from the sequence based on the iterations left.
    - __len__: Returns the total number of iterations left for all items.
    - get_resume_dict: Returns the updated resume_dict after completing the iterations.
    """

    def __init__(
        self,
        sequence: list[dict[str, Any]],
        iterations_per_record: int,
        seen_records: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """
        Initialize the enumerator with the given sequence, iterations_per_record, and an optional list of seen_records.
        """
        self.sequence = sequence
        self.iterations_per_record = iterations_per_record
        # Initialize resume_dict based on seen_records
        self.resume_dict = (
            Counter(record["question_id"] for record in seen_records)
            if seen_records
            else Counter()
        )
        # Calculate the total length based on the number of iterations left for each record
        self.length = sum(
            (iterations_per_record - self.resume_dict.get(record["question_id"], 0))
            for record in sequence
        )

    def __iter__(self) -> Iterator[dict]:
        """
        Yield each record based on the number of iterations left for that record.
        """
        idx = 0
        for item in self.sequence:
            question_id = item["question_id"]
            # Calculate the number of iterations left for the current record
            iterations_left = self.iterations_per_record - self.resume_dict.get(
                question_id, 0
            )
            for _ in range(iterations_left):
                # yield idx, item
                yield item
                idx += 1
            # Update resume_dict for the current record
            self.resume_dict[question_id] = self.iterations_per_record

    def __len__(self) -> int:
        """
        Return the total number of iterations left for all records.
        """
        return self.length

    def get_resume_dict(self) -> dict[str, int]:
        """
        Return the updated resume_dict after completing the iterations.
        """
        return dict(self.resume_dict)


class ResumeHandler:
    def __init__(
        self,
        records_from_previous_run: list[dict[str, Any]],
        iterations_per_question: int = 1,
    ):
        self.iterations_per_question = iterations_per_question
        self.counter: Counter = Counter()
        for record in records_from_previous_run:
            self.counter[record["question_id"]] += 1

    def should_skip_record(self, question_id: str) -> bool:
        if self.counter[question_id] >= self.iterations_per_question:
            return True
        return False

    def increment_iterations_for_record(self, question_id: str) -> None:
        self.counter[question_id] += 1


class JsonlIoHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def append_dict(self, data: dict) -> None:
        """Appends a dictionary as a new line in the JSONL file."""
        with open(self.file_path, "a") as f:
            json_str = json.dumps(data)
            f.write(json_str + "\n")

    def read_all(self, progress: Optional[bool] = False) -> list[dict]:
        """Reads all dictionaries from the JSONL file. Optional progress indicator."""
        data = []
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            if progress:
                lines = tqdm(lines, desc="Reading JSONL")
            for line in lines:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
        return data

    def read_n(self, n: int) -> list[dict]:
        """Reads the first n dictionaries from the JSONL file."""
        data = []
        with open(self.file_path, "r") as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                json_obj = json.loads(line.strip())
                data.append(json_obj)
        return data


class MeasureTime:
    def __enter__(self) -> "MeasureTime":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time


class InterceptHandler(logging.Handler):
    @logger.catch(default=True, onerror=lambda _: sys.exit(1))
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# We have a function specifically for this rather than
# using pure hydra.instantiate because we need to set
# the pad_token_id to the eos_token_id for llama models.
def init_llama_tokenizer(*args, **kwargs):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str)


def visualize_boxes_on_image(
    boxes: list[tuple[int, int, int, int]],
    image: Image.Image,
    categories: Optional[list[str]] = None,
    format: str = "xyxy",
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    """
    Visualizes multiple bounding boxes on the original image and annotates them with the category names if provided.
    Boxes can be in (left, upper, right, lower) format if format="lu" or (x, y, w, h) if format="xy".
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for i, box in enumerate(boxes):
        if format == "xyxy":
            left, upper, right, lower = box
        elif format == "xywh":
            x, y, w, h = box
            left, upper, right, lower = x, y, x + w, y + h
        else:
            raise ValueError("Invalid box format")

        # Draw the bounding box
        draw.rectangle([left, upper, right, lower], outline=color, width=width)  # type: ignore

        # Write the category name if provided
        if categories and categories[i]:
            font_size = 20
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            text_position = (left, upper - font_size)
            draw.text(text_position, categories[i], fill=color, font=font)

    return canvas


class ModuleProvider:
    _instance: Optional["ModuleProvider"] = None
    _modules: dict[str, Callable] = {}

    def __new__(cls: Any) -> Any:
        if cls._instance is None:
            cls._instance = super(ModuleProvider, cls).__new__(cls)
        return cls._instance

    def set_module(self, key: str, module: Any) -> None:
        """Set a module with a given key."""
        self._modules[key] = module

    def get_module(self, key: str) -> Optional[Any]:
        """Retrieve a module using its key. Returns None if the key does not exist."""
        return self._modules.get(key)

    def remove_module(self, key: str) -> None:
        """Remove a module using its key."""
        if key in self._modules:
            del self._modules[key]


def parse_outer_tag_of_pseudo_xml(xml_text: str) -> tuple[str, Optional[str]]:
    """
    Match a span of text enclosed in xml-like tags and return the tag name and the content.

    Parameters:
    ----------
    xml_text: str
        A string containing text enclosed in xml-like tags.

    Returns:
    -------
    tuple[str, Optional[str]]
        A tuple containing the tag name and the content enclosed in the tags.
    """
    # The function is written this way because parsing it in any other way
    # is hard. We can't use xml.etree.ElementTree because the input text
    # is not valid XML. So the approach I've taken is to use regex.
    pattern = r"<(?P<tag_name>\w+)>(?P<content>(?:.|\n|\r)*?)</(?P=tag_name)>"
    # There can be multiple matches, but we will only return the first one.
    matches = re.finditer(pattern, xml_text)
    try:
        match = next(matches)
    except StopIteration:
        raise ValueError("Provided XML is ill-formed")
    else:
        return match.group("tag_name"), match.group("content")


def remove_substring_having_xml_tag(text: str, tag: str) -> str:
    """
    Remove all text enclosed in a given XML tag from the input text.
    """
    pattern = re.compile(rf"<{tag}>.*?</{tag}>")
    return pattern.sub("", text)


def extract_python_code_in_triple_backticks(text: str) -> Optional[str]:
    pattern = r"```python\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


DEFAULT_RESTRICTED_BUILTINS = {
    "compile",
    "exec",
    "eval",
    "globals",
    "locals",
    "open",
    "input",
    "execfile",
    "__import__",
    "exit",
    "quit",
    "importlib",
}


def find_imports(code: str) -> list:
    """
    Identify import statements in the given Python code.

    Args:
        code (str): The Python code to analyze.

    Returns:
        list: A list of import statements found in the code.
    """
    import_statements = []

    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Traverse the AST to find import statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_statements.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name:
                for alias in node.names:
                    import_statements.append(f"{module_name}.{alias.name}")

    return import_statements


def find_not_allowed_functions(code: str, restricted_functions: set) -> list:
    """
    Identify not allowed function calls in the given Python code.

    Args:
        code (str): The Python code to analyze.
        restricted_functions (list): A list of functions not allowed to be called.

    Returns:
        list: A list of not allowed function calls found in the code.
    """
    not_allowed_functions = []

    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Traverse the AST to find function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                function_name = f"{node.func.value.id}.{node.func.attr}"
            if function_name and function_name in restricted_functions:
                not_allowed_functions.append(function_name)

    return not_allowed_functions


class SecurityException(Exception):
    pass


class ExecWithLimitedNamespace:
    def __init__(
        self,
        allowed_names: Optional[set[str]] = None,
        restricted_names: Optional[set[str]] = None,
        restricted_builtins: set[str] = DEFAULT_RESTRICTED_BUILTINS,
        inherited_scope: Optional[dict] = None,
    ):
        """
        This is a very janky way to get some security for the code we're running
        from the LLM. You can easily break out of this jail by doing Python tricks,
        but this is what I could whip up in a short time.

        Parameters
        -----------
        allowed_names: set[str]
            These are names that will explicitly be allowed in the namespace. For the
            visual programming environment, you want to give the agent access to `image`,
            `ImagePatch`, `bool_to_yesno`, and so on.
        restricted_names: set[str]
            These are function calls that are not allowed. We already have a mechanism to prevent using
            anything but allowed builtins and the whitelisted names in allowed_names, but this
            is an extra layer of security. We will check the ast to make sure none of these functions
            are called. The one I specifically want to disable is stuff like `get_ipython`, because it
            allows you to run shell commands.
        restricted_builtins: set[str]
            I've already given a reasonable set in DEFAULT_RESTRICTED_BUILTINS. This set is still "unsafe"
            because you can do stuff with getattr that will let you exec stuff. But I don't think the LLM
            will be doing anything like this.
        inherited_scope: Optional[dict]
            These are the variables from the enclosing scope. Anything from `allowed_names` will be inherited from
            the enclosing scope, while everything else will be inaccessible.
        """

        if restricted_names is None:
            self.restricted_names: set[str] = set()
        else:
            self.restricted_names = restricted_names

        self.restricted_builtins = restricted_builtins

        self.inherited_scope = inherited_scope or {}
        self.builtins = {
            k: v
            for k, v in builtins.__dict__.items()
            if k not in self.restricted_builtins
        }
        self.namespace = {}
        self.namespace.update(self.builtins)

        if allowed_names is not None:
            self.namespace.update(
                {k: v for k, v in self.inherited_scope.items() if k in allowed_names}
            )

    def __call__(self, code: str):
        imports = find_imports(code)
        not_allowed_functions = find_not_allowed_functions(
            code, self.restricted_builtins | self.restricted_names
        )
        if not_allowed_functions:
            raise SecurityException(
                f"""Your code used the following not allowed functions: {not_allowed_functions}.
Do not attempt to access the filesystem or network."""
            )
        if imports:
            raise SecurityException(
                "You are not allowed to use imports. Please use only the provided modules and functions."
            )
        bytecode = compile(code, filename="<string>", mode="exec")
        exec(bytecode, self.namespace, self.namespace)

    def serialize(self) -> str:
        namespace_to_repr = {
            k: repr(v)
            for k, v in self.namespace.items()
            if k not in self.builtins and k != "builtins" and not k.startswith("__")
        }
        return json.dumps(namespace_to_repr)
