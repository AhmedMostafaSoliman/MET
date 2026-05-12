import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any
import wandb


LOGGER = logging.getLogger(__name__)


class WandbLogger:
    """
    A generic Weights & Biases logger for research projects.

    This logger provides a flexible interface for tracking experiments, datasets,
    models, and predictions using Weights & Biases. It supports logging of
    hyperparameters, metrics, artifacts (datasets, models, etc.), and can be
    configured to operate in offline mode for environments without internet connectivity.

    Key Features:
        - Flexible initialization:  Accepts a dictionary of configuration parameters.
        - Offline mode:  Operates without internet connection, syncing data later.
        - Artifact management:  Logs datasets, models, and other artifacts.
        - Metric tracking:  Logs arbitrary metrics during training or evaluation.
        - Customizable logging frequency: Control how often metrics are logged.
        - Resuming runs:  Supports resuming previous W&B runs.
        - Project agnostic:  Can be used with any project.

    Usage:
        1. Initialize the logger with a configuration dictionary.
        2. Log hyperparameters using the `log_config` method.
        3. Log metrics during training or evaluation using the `log` method.
        4. Log artifacts (datasets, models) using the `log_artifact` method.
        5. Call `finish_run` at the end of the experiment to finalize the W&B run.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project: str = "default_project",
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        job_type: str = "Training",
        offline: bool = False,
        upload_dataset: bool = False,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[list] = None,
        id: Optional[str] = None,
        save_code: bool = True,
        log_model: bool = True,
        resume: str = "allow",
        enabled: bool = True,  # Add enabled flag
        **kwargs,
    ):  # Allow extra kwargs for future compatibility
        """
        Initializes the WandbLogger.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration parameters for the run.
            project (str): The name of the W&B project. Defaults to "default_project".
            entity (Optional[str]): The W&B entity (team or user).  Defaults to None.
            run_id (Optional[str]): The ID of the W&B run to resume. Defaults to None.
            job_type (str): The type of job (e.g., "Training", "Evaluation"). Defaults to "Training".
            offline (bool): Whether to run in offline mode. Defaults to False.
            upload_dataset (bool): Whether to upload the dataset as an artifact. Defaults to False.
            name (Optional[str]):  A descriptive name for the run.
            notes (Optional[str]):  Notes about the run.  Shows up in W&B UI.
            tags (Optional[list]):  Tags to categorize the run in W&B.
            id (Optional[str]):  Explicit run ID. Useful for resuming and linking runs.
            save_code (bool):  Whether to save the executed code to W&B.
            log_model (bool): Whether to automatically log the model as an artifact.
            resume (str):  "allow" to resume if possible, "must" to resume, None/False to disable.
            enabled (bool): Whether WandB logging is enabled.
            **kwargs: Additional keyword arguments.

        Raises:
            ImportError: If wandb is not installed and not running in offline mode.
        """

        self.config = config
        self.project = project
        self.entity = entity
        self.run_id = run_id
        self.job_type = job_type
        self.offline = offline
        self.upload_dataset = upload_dataset
        self.name = name
        self.notes = notes
        self.tags = tags
        self.id = id
        self.save_code = save_code
        self.log_model = log_model
        self.resume = resume
        self.enabled = enabled  # Store enabled flag

        self.wandb, self.wandb_run = wandb, None
        self.log_dict: Dict[str, Any] = {}
        self.artifacts: Dict[str, wandb.Artifact] = {}  # Store artifacts for later use.
        self.tables: Dict[str, wandb.Table] = {}  # Store tables.

        if not self.enabled:
            if LOGGER:
                LOGGER.info("WandB logging is disabled")
            return

        if self.wandb is None and not self.offline:
            raise ImportError("Weights & Biases is not installed. Please install it or run in offline mode.")

        try:
            self.wandb_run = (
                wandb.init(
                    config=self.config,
                    project=self.project,
                    entity=self.entity,
                    id=self.id,
                    resume=self.resume,
                    job_type=self.job_type,
                    name=self.name,
                    notes=self.notes,
                    tags=self.tags,
                    allow_val_change=True,
                    sync_tensorboard=False,  # Disable TensorBoard syncing
                    save_code=self.save_code,
                    mode="offline" if self.offline else "online",
                    **kwargs,
                )
                if not wandb.run
                else wandb.run
            )
            if LOGGER:
                LOGGER.info(f"W&B Run initialized with ID: {self.wandb_run.id}")  # type: ignore
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"Failed to initialize W&B run: {e}")  # type: ignore
            self.wandb_run = None

    def log_config(self, config: Dict[str, Any]):
        """Logs the configuration parameters to W&B."""
        if self.wandb_run and self.enabled:
            self.wandb_run.config.update(config, allow_val_change=True)

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        """
        Logs metrics to W&B.

        Args:
            log_dict (Dict[str, Any]): A dictionary of metrics to log.
            step (Optional[int]):  The step number for the metrics (e.g., epoch or iteration).
        """
        if self.wandb_run and self.enabled:
            try:
                if step is not None:
                    wandb.log(log_dict, step=step)
                else:
                    wandb.log(log_dict)
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error logging to W&B: {e}")  # type: ignore

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str,
        aliases: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_as_training_data: bool = False,
        use_as_model: bool = False,
        description: Optional[str] = None,
    ):
        """
        Logs an artifact (e.g., dataset, model) to W&B.

        Args:
            artifact_path (str): The path to the artifact file or directory.
            artifact_name (str): The name of the artifact.
            artifact_type (str): The type of the artifact (e.g., "dataset", "model").
            aliases (Optional[list]): A list of aliases for the artifact. Defaults to None.
            metadata (Optional[Dict[str, Any]]): A dictionary of metadata for the artifact. Defaults to None.
            use_as_training_data (bool): if the artifact should be added to the W&B Input Artifact.
            use_as_model (bool): if the artifact should be added to the W&B Output Artifact.
            description (Optional[str]): A description for this artifact.
        """
        if self.wandb_run and self.enabled:
            try:
                artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata, description=description)
                artifact.add_file(artifact_path)  # or artifact.add_dir()

                # If the artifact already exists, we'll overwrite it
                # with the new version.
                if artifact_name in self.artifacts:
                    old_artifact = self.artifacts[artifact_name]
                    self.wandb_run.log_artifact(old_artifact, aliases=["obsolete"])

                # Log the new artifact.
                self.wandb_run.log_artifact(artifact, aliases=aliases)
                self.artifacts[artifact_name] = artifact
                if LOGGER:
                    LOGGER.info(f"Logged artifact '{artifact_name}' of type '{artifact_type}' from '{artifact_path}'")  # type: ignore
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error logging artifact: {e}")  # type: ignore

    def log_table(self, table_name: str, data: list, columns: list, aliases: Optional[list] = None):
        """Logs a W&B Table.

        Args:
            table_name: The name of the table.
            data: A list of rows to add to the table.
            columns: A list of column names.
            aliases: A list of aliases for this table.
        """
        if self.wandb_run and self.enabled:
            try:
                table = wandb.Table(data=data, columns=columns)
                wandb.log({table_name: table})
                # self.tables[table_name] = table
                if aliases:
                    for alias in aliases:
                        wandb.log({f"{table_name}_{alias}": table})
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error logging table to W&B: {e}")  # type: ignore

    def log_summary(self, summary_dict: Dict[str, Any]):
        """Logs summary metrics that will appear in the final summary."""
        if self.wandb_run and self.enabled:
            try:
                for key, value in summary_dict.items():
                    self.wandb_run.summary[key] = value
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error logging summary to W&B: {e}")  # type: ignore

    def finish_run(self):
        """Finishes the W&B run."""
        if self.wandb_run and self.enabled:
            try:
                self.wandb_run.finish()
                if LOGGER:
                    LOGGER.info("Finished W&B run")  # type: ignore
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error finishing W&B run: {e}")  # type: ignore

    def watch(self, models, log="gradients", log_freq=1000):
        """Enable gradient logging for a model"""
        if self.wandb_run and self.enabled:
            wandb.watch(models, log=log, log_freq=log_freq)

    def define_metric(self, name, step_metric="epoch"):
        """Define custom metric axes"""
        if self.wandb_run and self.enabled:
            wandb.define_metric(name, step_metric=step_metric)

    def alert(self, title: str, text: str, level: str = "INFO"):
        """Send an alert to W&B"""
        if self.wandb_run and self.enabled:
            try:
                wandb.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO))
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Error sending alert to W&B: {e}")  # type: ignore
