🎯 Core Philosophy: "Simple & Sweet"
Prioritize code that can be understood by a junior developer in 10 seconds over "clever" or "compact" code. If a solution is too complex to explain verbally, it is too complex for the codebase.

1. Complexity Control
No Deep Nesting: Use Guard Clauses (early returns) to keep the main logic at the lowest indentation level possible.

Flat is better than nested: Avoid deeply nested loops or conditionals.

Function Length: Aim for functions under 20 lines. If a function performs multiple steps, break it into smaller, descriptive helper functions.

Standard Library First: Prefer Python’s built-in functions and standard library over external dependencies or custom-built abstractions unless absolutely necessary.

Avoid Cleverness: No complex list comprehensions with multiple if/else logic. Use a standard for loop if it improves readability.

2. Mandatory Naming Conventions
Every name must act as documentation. Do not use abbreviations.

Variables & Constants
Format: snake_case for variables, UPPER_SNAKE_CASE for constants.

Descriptive: Never use data, item, val, or temp.

Booleans: Must start with a "status" verb: is_valid, has_token, should_refresh, can_execute.

Collections: Use plural nouns for lists/dicts (e.g., user_ids, config_map).

Functions & Classes
Format: snake_case for functions, PascalCase for classes.

Verb-Noun Pairs: Functions must start with a verb (e.g., get_user_email(), process_payment_payload(), validate_session_expiry()).

Internal Methods: Prefix strictly internal class methods with a single underscore (e.g., _format_internal_date()).

3. Implementation Standards
Type Hinting: Mandatory for all function arguments and return types. Use list[str] instead of List[str] (Python 3.10+ style).

F-Strings Only: Use f-strings for all string interpolations. No .format() or % operator.

Error Handling: Use specific exceptions. Never use except Exception: pass. Always log the error or handle it gracefully.

Pathing: Use pathlib for all file system operations. Avoid os.path.

4. Documentation Style
Google Style Docstrings: Provide a brief summary, Args, and Returns for any non-trivial function.

Self-Documenting Code: Comments should explain "Why" something is done, not "What" is being done (the code should make the "What" obvious).

🛑 Negative Patterns (DO NOT SUGGEST)
No Lambda functions for complex logic.

No **kwargs or *args unless specifically required by a library/framework; prefer explicit arguments.

No global variables.

No "Magic Numbers"; use named constants.