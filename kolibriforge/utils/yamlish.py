"""Minimal YAML-subset parser (zero-dependency fallback).

Only the subset used by config files is supported: flat mappings, nested
mappings, block lists (`- item`), inline lists (`[a, b]`), scalars and inline
comments. This keeps the CLI able to read .yaml configs on a managed Python
that does not ship PyYAML.
"""
import re


def parse(text):
    lines = [ln.rstrip() for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    root = {}
    stack = [(-1, root, None)]  # (indent, container, key_in_parent)
    for ln in lines:
        indent = len(ln) - len(ln.lstrip(" "))
        content = ln.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        if content.startswith("- "):
            rest = content[2:].strip()
            parent = stack[-1][1]
            if isinstance(parent, dict):
                gp = stack[-2][1]
                gk = stack[-1][2]
                if isinstance(gp, dict):
                    gp[gk] = []
                else:
                    gp.append([])
                stack[-1] = (stack[-1][0], gp[gk], gk)
                parent = gp[gk]
            if ":" in rest:
                key, _, val = rest.partition(":")
                key = key.strip()
                val = _strip_comment(val).strip()
                if val == "":
                    child = {}
                    parent.append(child)
                    stack.append((indent, child, key))
                else:
                    parent.append({key: _scalar(val)})
            else:
                parent.append(_scalar(rest))
            continue
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip().strip('"').strip("'")
        val = _strip_comment(val).strip()
        parent = stack[-1][1]
        if val == "":
            child = {}
            if isinstance(parent, dict):
                parent[key] = child
            else:
                parent.append(child)
            stack.append((indent, child, key))
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            items = [] if not inner else [x.strip().strip('"').strip("'") for x in inner.split(",")]
            if isinstance(parent, dict):
                parent[key] = items
            else:
                parent.append(items)
        else:
            if isinstance(parent, dict):
                parent[key] = _scalar(val)
            else:
                parent.append(_scalar(val))
    return root


def _strip_comment(val):
    if " #" in val:
        return val[:val.index(" #")]
    return val


def _scalar(v):
    v = v.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "None", "~"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v
