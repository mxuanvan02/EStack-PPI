from colorama import Fore, Style
import pandas as pd


class PipelineLogger:
    """Lightweight logger for experiment phases and metric tables."""

    def header(self, title: str):
        print("\n" + "=" * 80)
        print(f"🚀 {Fore.CYAN}{Style.BRIGHT}{title.upper()}{Style.RESET_ALL}")
        print("=" * 80)

    def phase(self, phase_name: str):
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}📦 PHASE: {phase_name.upper()}{Style.RESET_ALL}")
        print("-" * 60)

    def info(self, message: str):
        print(f"  {Fore.WHITE}ℹ️  {message}")

    def result(self, key: str, value: any, indent: int = 2):
        indent_space = " " * indent
        print(f"{indent_space}{Fore.GREEN}✅ {key:<25}: {Style.BRIGHT}{value}{Style.RESET_ALL}")

    def warning(self, message: str):
        print(f"  {Fore.YELLOW}⚠️  {message}")

    def metric_table(self, df: pd.DataFrame, title: str):
        print(f"\n--- {title} ---")
        print(df.to_string())
