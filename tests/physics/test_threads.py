import os

from reservoir_backend.eos.threads import cap_flash_threads, ensemble_flash_threads


def test_ensemble_flash_threads_default_is_one() -> None:
    prev = os.environ.get("RESERVOIR_FLASH_THREADS")
    os.environ.pop("RESERVOIR_FLASH_THREADS", None)
    try:
        assert ensemble_flash_threads() == 1
        cap_flash_threads(2)
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["NUMBA_NUM_THREADS"] == "2"
    finally:
        if prev is None:
            os.environ.pop("RESERVOIR_FLASH_THREADS", None)
        else:
            os.environ["RESERVOIR_FLASH_THREADS"] = prev
