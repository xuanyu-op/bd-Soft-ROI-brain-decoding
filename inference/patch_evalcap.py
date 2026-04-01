import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr

_PATCHED = False


@contextmanager
def silence_evalcap():
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def apply_patches():
    global _PATCHED
    if _PATCHED:
        return

    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer as OriginalPTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu as OriginalBleu

    _original_bleu_compute_score = OriginalBleu.compute_score

    class QuietPTBTokenizer(OriginalPTBTokenizer):
        def tokenize(self, *args, **kwargs):
            with silence_evalcap():
                return super().tokenize(*args, **kwargs)

    def _quiet_bleu_compute_score(self, *args, **kwargs):
        with silence_evalcap():
            return _original_bleu_compute_score(self, *args, **kwargs)

    import pycocoevalcap.tokenizer.ptbtokenizer
    pycocoevalcap.tokenizer.ptbtokenizer.PTBTokenizer = QuietPTBTokenizer

    import pycocoevalcap.bleu.bleu
    pycocoevalcap.bleu.bleu.Bleu.compute_score = _quiet_bleu_compute_score

    _PATCHED = True