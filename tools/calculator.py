"""Restricted arithmetic calculator; intentionally does not use eval()."""
import ast
import operator

_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _calculate(node):
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large.")
        result = _ALLOWED[type(node.op)](left, right)
        if abs(result) > 1e100:
            raise ValueError("Result is too large.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_calculate(node.operand))
    raise ValueError("Only basic arithmetic expressions are allowed.")

def calculator(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _calculate(tree)
        return {"result": str(result), "value": result}
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError) as exc:
        return {"result": f"Calculator error: {exc}", "error": str(exc)}
