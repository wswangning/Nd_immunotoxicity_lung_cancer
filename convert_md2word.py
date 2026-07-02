import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from scripts.md2word import main
import sys as sys_mod

# Replace emoji characters in the script
import scripts.md2word as md2word_mod
original_main = md2word_mod.main

def patched_main():
    # Monkey patch print to handle emoji
    import builtins
    original_print = builtins.print
    def safe_print(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = arg.replace('\U0001f4cb', '[INFO]').replace('\u2705', '[OK]').replace('\u26a0\ufe0f', '[WARN]').replace('\u274c', '[ERR]')
            new_args.append(arg)
        original_print(*new_args, **kwargs)
    builtins.print = safe_print
    try:
        original_main()
    finally:
        builtins.print = original_print

md2word_mod.main = patched_main

if __name__ == '__main__':
    patched_main()