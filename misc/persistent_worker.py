# persistent_worker.py
"""
Persistent worker process + manager with watchdog restart.

Worker process:
  - Loads HFBackend once (heavy initialization).
  - Listens on request_q for (req_id, payload).
  - For each generation request, spawns a thread that calls HFBackend.generate(...)
  - The worker joins the thread with the requested max_time (seconds). If the thread
    finishes, result is put onto response_q. If it does not finish within max_time,
    worker logs and self-terminates (os._exit) to ensure GPU memory is freed.
    The manager will detect the worker died and will respawn it.

Manager (PersistentWorkerManager):
  - Spawn worker process and queues.
  - Provides generate(prompt, timeout=...) that puts request on queue and waits for
    response or worker death. If worker dies, it will respawn and return a timeout.
"""

import multiprocessing as mp
from multiprocessing.synchronize import Event
import threading
import time
import os
import traceback
from typing import Dict, Any
from llm_backends import HFBackend

# Timeout for queue.get in manager loop (seconds)
_QUEUE_GET_POLL = 0.5
def _worker_main(request_q: mp.Queue, response_q: mp.Queue, model_source: str, quant_4bit: bool, init_event: Event):
    """
    Loads HFBackend and serves requests until killed.
    Each request payload: dict {
        "prompt": str,
        "max_new_tokens": int,
        "temperature": float,
        "top_p": float,
        "do_sample": bool,
        "max_time": float or None,
        "batch_prompts": optional list[str]
    }
    Response: put (req_id, {"ok": True/False, "text": "...", "error": "..."})
    If a generation hangs beyond max_time, the worker will os._exit(3) to force termination.
    """
    try:
        # Initialize backend (loads model once)
        hb = HFBackend(local_model_path=os.environ.get("LOCAL_MODEL_PATH", None),
                       hf_model_id=model_source,
                       quantize_4bit=quant_4bit)
    except Exception as e:
        tb = traceback.format_exc()
        try:
            response_q.put(("__worker_init_failed__", {"ok": False, "error": f"Worker init failed: {e}", "trace": tb}))
        except Exception:
            pass
        try:
            init_event.set()
        except Exception:
            pass
        os._exit(2)

    try:
        init_event.set()
    except Exception:
        pass

    # serve loop
    while True:
        try:
            # block until request arrives
            req_id, payload = request_q.get()
        except Exception:
            break

        if req_id == "__shutdown__":
            try:
                response_q.put(("__shutdown__", {"ok": True}))
            except Exception:
                pass
            break

        # unpack payload
        prompt = payload.get("prompt")
        max_new_tokens = int(payload.get("max_new_tokens", 256))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        do_sample = bool(payload.get("do_sample", False))
        max_time = payload.get("max_time", None)  # seconds float or None
        batch_prompts = payload.get("batch_prompts", None)

        # thread target to run generation
        result_container: Dict[str, Any] = {}

        def _gen_call():
            try:
                if batch_prompts:
                    if hasattr(hb, "generate_batch"):
                        texts = hb.generate_batch(batch_prompts,
                                                  max_new_tokens=max_new_tokens,
                                                  temperature=temperature,
                                                  max_time=max_time)
                        # encode as a single string joined by delimiter (manager will pass through)
                        result_container["ok"] = True
                        result_container["text"] = texts
                    else:
                        # fallback: sequentially generate
                        res_list = []
                        for p in batch_prompts:
                            r = hb.generate(p, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample, max_time=max_time)
                            res_list.append(r)
                        result_container["ok"] = True
                        result_container["text"] = res_list
                else:
                    text = hb.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample, max_time=max_time)
                    result_container["ok"] = True
                    result_container["text"] = text
            except Exception as e:
                result_container["ok"] = False
                result_container["error"] = f"Generation failed: {repr(e)}\n{traceback.format_exc()}"

        gen_thread = threading.Thread(target=_gen_call, daemon=True)
        gen_thread.start()

        # join with watchdog timeout
        start = time.time()
        if max_time is None:
            try:
                max_time = float(os.environ.get("LLM_WORKER_MAX_TIME", "30"))
            except Exception:
                max_time = 30.0

        # give a small grace interval for cleanup (in seconds)
        grace = float(os.environ.get("LLM_WORKER_GRACE_SEC", "5.0"))
        gen_thread.join(timeout=max_time + grace)
        if gen_thread.is_alive():
            try:
                response_q.put((req_id, {"ok": False, "error": f"generation timeout (worker killed after {max_time + grace}s)"}))
            except Exception:
                pass
            # flush logs then force exit: ensures model/inference is killed and GPU memory freed
            os._exit(3)
        else:
            # thread finished; put result to response queue
            try:
                if result_container.get("ok"):
                    response_q.put((req_id, {"ok": True, "text": result_container.get("text")}))
                else:
                    response_q.put((req_id, {"ok": False, "error": result_container.get("error", "Unknown generation error")}))
            except Exception:
                pass

    os._exit(0)


class PersistentWorkerManager:
    """
    Class used by client process to talk with worker process via queues.
    - It will spawn the worker process and automatically restart it if it dies.
    - generate(prompt, timeout=client_wait_seconds, **payload_args) will:
        * put (req_id, payload) on request_q
        * wait up to `timeout` seconds for corresponding response
        * If worker dies mid-request, manager restarts worker and returns timeout result.
    """
    def __init__(self, model_source: str, quant_4bit: bool = True):
        self.model_source = model_source
        self.quant_4bit = quant_4bit
        self._start_worker()

    def _start_worker(self):
        self.request_q = mp.Queue()
        self.response_q = mp.Queue()
        self.init_event = mp.Event()
        self.proc = mp.Process(target=_worker_main, args=(self.request_q, self.response_q, self.model_source, self.quant_4bit, self.init_event), daemon=True)
        self.proc.start()
        # wait for worker initialization or fail
        started = self.init_event.wait(timeout=60.0)
        if not started:
            # worker failed to init in reasonable time -> try to read any init error from response_q
            try:
                while not self.response_q.empty():
                    rid, payload = self.response_q.get_nowait()
                    if rid == "__worker_init_failed__":
                        raise RuntimeError(payload.get("error", "worker init failed"))
            except Exception:
                pass
            raise RuntimeError("Worker failed to initialize within 60s.")

    def _restart_worker(self):
        try:
            # attempt graceful shutdown
            try:
                self.request_q.put(("__shutdown__", {}), block=False)
            except Exception:
                pass
            # terminate process if alive
            if self.proc.is_alive():
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        except Exception:
            pass

        self._start_worker()

    def generate(self, prompt: str, timeout: float = 30.0, **payload) -> Dict[str, Any]:
        """
        Put request on worker queue and wait for response up to `timeout` seconds.
        `payload` may include: max_new_tokens, temperature, top_p, do_sample, max_time, batch_prompts
        Returns dict: {"ok": True, "text": ...} or {"ok": False, "error": "..."}
        """
        if not self.proc.is_alive():
            # try to restart
            try:
                self._restart_worker()
            except Exception as e:
                return {"ok": False, "error": f"worker not alive and restart failed: {e}"}

        req_id = f"{time.time()}-{os.getpid()}-{threading.get_ident()}"
        full_payload = dict(payload)
        full_payload["prompt"] = prompt

        try:
            self.request_q.put((req_id, full_payload), block=False)
        except Exception:
            # if queue failed, try restart once
            if not self.proc.is_alive():
                try:
                    self._restart_worker()
                except Exception:
                    return {"ok": False, "error": "failed to put request; worker restart failed"}
                try:
                    self.request_q.put((req_id, full_payload), block=False)
                except Exception:
                    return {"ok": False, "error": "failed to enqueue request after restart"}
            else:
                return {"ok": False, "error": "failed to enqueue request"}

        # wait for response (poll because worker may die)
        start = time.time()
        while True:
            # if worker died, break and return timeout
            if not self.proc.is_alive():
                try:
                    self._restart_worker()
                except Exception:
                    pass
                return {"ok": False, "error": "worker died during generation (timed out on client side)"}

            # attempt to get response without blocking too long
            try:
                rid, payload = self.response_q.get(timeout=_QUEUE_GET_POLL)
            except Exception:
                if time.time() - start > timeout:
                    return {"ok": False, "error": f"client wait timeout after {timeout}s"}
                continue

            if rid == req_id:
                return payload
            elif rid == "__shutdown__":
                # ignore or requeue?
                continue
            else:
                # responses for other requests (unlikely); ignore
                continue

    def shutdown(self):
        try:
            if self.proc.is_alive():
                self.request_q.put(("__shutdown__", {}))
                self.proc.join(timeout=3.0)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass
