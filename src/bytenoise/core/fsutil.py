"""Small filesystem helpers shared by every writer."""
import os
import time


def replace_retry(src, dst, tries=40, delay=0.05):
    """os.replace that tolerates Windows' transient PermissionError while a reader has the destination open."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def atomic_write(path, text):
    """Write via a process-unique temp file in the same directory, then replace."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_retry(tmp, path)
