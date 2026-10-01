"""
Calculator Tool using simpleeval for safe mathematical evaluations.
Assignment 7 requirement 2: Add a calculator tool (using simpleeval) for arithmetic
the model shouldn't do in its head.
"""

import math
from typing import Dict, Any
from simpleeval import SimpleEval, FunctionNotDefined, NameNotDefined, FeatureNotAvailable


# Standard math functions safely exposed to simpleeval
MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
}

# Common mathematical constants
MATH_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


def calculator(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate an arithmetic expression using simpleeval.

    Args:
        expression: A mathematical expression string, e.g. "155 - 43", "(120 + 74) / 2", "sqrt(144) * 5".

    Returns:
        Dict with "result", "value", and "expression" or "error".
    """
    if expression is None or str(expression).strip() == "":
        return {
            "status": "error",
            "error": "Expression cannot be empty.",
            "result": "Please provide an arithmetic expression."
        }

    clean_expr = str(expression).strip()

    evaluator = SimpleEval()
    evaluator.functions = dict(MATH_FUNCTIONS)
    evaluator.names = dict(MATH_NAMES)

    try:
        raw_result = evaluator.eval(clean_expr)
        
        # Round floating point to avoid tiny floating point representation errors
        if isinstance(raw_result, float):
            result_val = round(raw_result, 6)
            if result_val.is_integer():
                result_val = int(result_val)
        else:
            result_val = raw_result

        return {
            "status": "success",
            "expression": clean_expr,
            "value": result_val,
            "result": str(result_val)
        }
    except ZeroDivisionError:
        return {"status": "error", "error": "Division by zero", "result": "Error: Division by zero"}
    except (FunctionNotDefined, NameNotDefined, FeatureNotAvailable, Exception) as exc:
        return {
            "status": "error",
            "error": f"Calculation error: {type(exc).__name__}: {exc}",
            "result": f"Could not evaluate expression '{clean_expr}': {exc}"
        }

