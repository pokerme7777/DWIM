import argparse
import ast


def strip_function_bodies_with_ast(source_code: str) -> str:
    """
    Strips the bodies of functions and classes using AST, leaving only stubs and
    docstrings.
    """
    parsed_ast = ast.parse(source_code)
    output_lines: list[str] = []

    def process_node(node, indent_level=0):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Generate function or class header
            header = " ".join(
                [
                    "async" if isinstance(node, ast.AsyncFunctionDef) else "",
                    "def" if not isinstance(node, ast.ClassDef) else "class",
                    node.name,
                ]
            ).strip()

            # Generate arguments for functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args]
                header += f"({', '.join(args)})"

            # Add colon and indent
            header += ":"
            output_lines.append(" " * indent_level + header)

            # Add docstring if exists
            docstring = ast.get_docstring(node)
            if docstring:
                formatted_docstring = (
                    '"""'
                    + docstring.replace("\n", "\n" + " " * (indent_level + 4))
                    + '"""'
                )
                output_lines.append(" " * (indent_level + 4) + formatted_docstring)

            # Add pass statement
            output_lines.append(" " * (indent_level + 4) + "pass")

        for child in ast.iter_child_nodes(node):
            process_node(
                child,
                indent_level + 4
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else indent_level,
            )

    process_node(parsed_ast)
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Strip function and class bodies from Python source code."
    )
    parser.add_argument(
        "source_file", type=str, help="Path to the Python source file to be stripped."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to save the stripped Python source code."
    )

    args = parser.parse_args()

    with open(args.source_file, "r") as f:
        source_code = f.read()

    stripped_code = strip_function_bodies_with_ast(source_code)

    with open(args.output_file, "w") as f:
        f.write(stripped_code)


if __name__ == "__main__":
    main()
