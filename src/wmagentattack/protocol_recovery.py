"""Conservative serialization recovery: explicit names/JSON only, no inference."""
import json
import re


def explicit_call(text, allowed_functions, validate_arguments):
    """Recover one whole-message call, leaving names and argument values intact.

    The validator is supplied by the runtime and must reject unknown keys and
    invalid types. No prose-to-call inference, eval, value coercion or defaults.
    """
    source = text.strip()
    if source.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", source, re.DOTALL)
        if match is None:
            return None
        source = match.group(1).strip()
    name, arguments = None, None
    if source.startswith("{"):
        try:
            payload = json.loads(source)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        name_keys = set(payload) & {"name", "function"}
        arg_keys = set(payload) & {"parameters", "arguments"}
        if len(name_keys) != 1 or len(arg_keys) != 1 or len(payload) != 2:
            return None
        name, arguments = payload[next(iter(name_keys))], payload[next(iter(arg_keys))]
    elif source.startswith("<function") and source.endswith("</function>"):
        body = source[:-len("</function>")].strip()
        # Allow an omitted '>' before the closing tag, but never a second call
        # or trailing explanation. JSONDecoder handles nested objects/strings.
        prefix = re.match(r"^<function\s*(?:=\s*name\s*=\s*|name\s*=\s*|=\s*|>\s*)([\"']?)([A-Za-z_]\w*)\1", body)
        if prefix is None:
            return None
        name = prefix.group(2)
        rest = body[prefix.end():].strip()
        rest = re.sub(r"^(?:parameters|arguments)\s*=\s*", "", rest).strip()
        if rest.startswith(">") or rest.startswith("("):
            rest = rest[1:].strip()
        if not rest.startswith("{"):
            return None
        try:
            arguments, end = json.JSONDecoder().raw_decode(rest)
        except ValueError:
            return None
        if rest[end:].strip() not in {"", ">", ")", ")>"}:
            return None
    if not isinstance(name, str) or name not in allowed_functions or not isinstance(arguments, dict):
        return None
    if not validate_arguments(name, arguments):
        return None
    return {"function": name, "args": arguments}


def valid_runtime_arguments(runtime, name, arguments):
    schema = runtime.functions[name].parameters
    if set(arguments) - set(schema.model_fields):
        return False
    try:
        schema.model_validate(arguments, strict=True)
    except (ValueError, TypeError):
        return False
    return True
