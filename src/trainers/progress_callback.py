from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from tqdm.auto import tqdm
from typing import Optional


class TqdmProgressCallback(TrainerCallback):
    """
    Lightweight tqdm progress indicator for Trainer runs. The built-in progress bar can be
    suppressed depending on the environment or logging setup; this callback ensures a visual
    indicator is always shown.
    """

    def __init__(self) -> None:
        self.progress_bar: Optional[tqdm] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.progress_bar is None:
            total = state.max_steps if state.max_steps and state.max_steps > 0 else None
            self.progress_bar = tqdm(
                total=total,
                desc="Training",
                dynamic_ncols=True,
                leave=True,
            )

        # Align the bar in case of resume-from-checkpoint
        if self.progress_bar is not None:
            self.progress_bar.n = state.global_step
            self.progress_bar.refresh()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update(state.global_step - self.progress_bar.n)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ) -> None:
        if self.progress_bar is not None and logs:
            loss = logs.get("loss")
            if loss is not None:
                self.progress_bar.set_postfix(loss=f"{loss:.4f}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

