from typing import Optional, Union
from statistics import mean
import lm_eval
import transformers

# Try different import paths for HFLM (modern replacement of HFModel)
try:
    from lm_eval.models.huggingface import HFLM
except ImportError:
    try:
        from lm_eval.models import HFLM
    except ImportError:
        try:
            from lm_eval import models
            HFLM = models.HFLM
        except Exception:
            raise ImportError(
                "Could not import HFLM from lm_eval. Your lm-eval-harness version may be incompatible."
            )

# simple_evaluate import attempts
try:
    from lm_eval.api import simple_evaluate
except ImportError:
    try:
        from lm_eval.evaluator import simple_evaluate
    except ImportError:
        raise ImportError("lm_eval.simple_evaluate not found in this installation")


class HarnessEvaluator:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
            transformers.PreTrainedTokenizer,
            transformers.PreTrainedTokenizerFast,
        ],
        tasks: Union[list[str], str],
        num_fewshot: Optional[int] = 0,
        batch_size: Optional[int] = 6,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = "auto",
        limit: Optional[Union[int, float]] = None,
        cache_requests: bool = True,
        apply_chat_template: bool = False,
        random_seed: int = 42,
        numpy_random_seed: int = 42,
        torch_random_seed: int = 42,
        fewshot_random_seed: int = 42,
    ) -> None:

        # Handle PreTrainedTokenizerFast case
        if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            tokenizer_name = None

            if hasattr(tokenizer, "name_or_path"):
                tokenizer_name = tokenizer.name_or_path
            elif hasattr(model, "name_or_path"):
                tokenizer_name = model.name_or_path
            elif hasattr(model, "config"):
                if hasattr(model.config, "_name_or_path"):
                    tokenizer_name = model.config._name_or_path
                elif hasattr(model.config, "name_or_path"):
                    tokenizer_name = model.config.name_or_path

            if tokenizer_name:
                try:
                    self.lm = HFLM(
                        model,
                        tokenizer=tokenizer_name,
                        batch_size=batch_size,
                        trust_remote_code=True,
                    )
                except Exception:
                    self.lm = HFLM(
                        model,
                        batch_size=batch_size,
                        trust_remote_code=True,
                    )
            else:
                self.lm = HFLM(
                    model,
                    batch_size=batch_size,
                    trust_remote_code=True,
                )

        else:
            # Regular tokenizer
            self.lm = HFLM(
                model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                trust_remote_code=True,
            )

        # Convert comma-delimited task string
        if isinstance(tasks, str):
            tasks = tasks.split(",")

        # utility pseudo-task
        self.utility_tasks = None
        if "utility" in tasks:
            self.utility_tasks = [
                "boolq",
                "rte",
                "hellaswag",
                "winogrande",
                "arc_challenge",
                "openbookqa",
            ]
            tasks.remove("utility")
            tasks.extend(self.utility_tasks)

        self.tasks = tasks
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.limit = limit
        self.device = device
        self.cache_requests = cache_requests
        self.random_seed = random_seed
        self.numpy_random_seed = numpy_random_seed
        self.torch_random_seed = torch_random_seed
        self.fewshot_random_seed = fewshot_random_seed
        self.apply_chat_template = apply_chat_template

    def run(self) -> dict:
        clean_results = {}

        eval_args = dict(
            model=self.lm,
            batch_size=self.batch_size,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            device=self.device,
            limit=self.limit,
        )

        import inspect
        valid_params = inspect.signature(simple_evaluate).parameters
        if "max_batch_size" in valid_params:
            eval_args["max_batch_size"] = self.max_batch_size

        results = simple_evaluate(**eval_args)["results"]

        for task in self.tasks:
            if task == "wmdp":
                for t in ["wmdp-bio"]:
                    clean_results[t] = results[t]["acc,none"]
                continue

            if self.utility_tasks is not None and task in self.utility_tasks:
                continue

            clean_results[task] = results[task]["acc,none"]

        if self.utility_tasks is not None:
            avg = mean(results[t]["acc,none"] for t in self.utility_tasks)
            clean_results["utility"] = avg

        return clean_results
