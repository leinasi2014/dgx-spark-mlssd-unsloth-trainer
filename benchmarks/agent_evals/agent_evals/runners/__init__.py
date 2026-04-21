from .mcpmark import run as run_mcpmark
from .swebench import run_multilingual, run_verified
from .terminal_bench import run as run_terminal_bench

RUNNERS = {
    "swebench_verified": run_verified,
    "swebench_multilingual": run_multilingual,
    "terminal_bench": run_terminal_bench,
    "mcpmark": run_mcpmark,
}
