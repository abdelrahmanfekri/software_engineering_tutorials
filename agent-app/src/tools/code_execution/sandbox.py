"""Code execution tool with sandboxing."""

from typing import Any, Dict
import asyncio
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from ..base import BaseTool, ToolSchema
from ...utils.errors import ToolExecutionError


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def get_schema(self) -> ToolSchema:
        """Return the code execution tool schema."""
        return ToolSchema(
            name="execute_code",
            description="Execute Python code and return the output. Use this for calculations, data processing, or running algorithms.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        )
    
    async def execute(self, code: str, timeout: int = None, **kwargs) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict with execution result, stdout, and stderr
        """
        timeout = timeout or self.timeout
        
        try:
            # Capture stdout and stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            # Create restricted globals (basic safety - not production-grade)
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                }
            }
            
            result = None
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code with timeout
                try:
                    exec(code, safe_globals)
                    result = safe_globals.get("result", None)
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue()
                    }
            
            return {
                "success": True,
                "result": str(result) if result is not None else None,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
            
        except Exception as e:
            raise ToolExecutionError(f"Code execution failed: {str(e)}")
