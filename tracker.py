import json
import time
from pathlib import Path
import config

class ExperimentTracker:
    """Experiment tracking manager that logs run parameters and metrics to structured JSON & MLflow."""
    
    def __init__(self, experiment_name="MediScan_Experiments"):
        self.experiment_name = experiment_name
        self.log_file = config.LOGS_DIR / "experiment_runs.json"
        self._ensure_log_file()
        
        # Try initializing MLflow if available
        self.mlflow_available = False
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            self.mlflow_available = True
        except Exception:
            self.mlflow_available = False
            
    def _ensure_log_file(self):
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump([], f, indent=2)
                
    def log_run(self, run_id, params, metrics, history=None):
        """Log a complete training or evaluation run."""
        run_data = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "metrics": metrics,
            "history": history or {}
        }
        
        # Save to local JSON database
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
            
        logs.append(run_data)
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)
            
        # Log to MLflow if enabled
        if self.mlflow_available:
            try:
                import mlflow
                with mlflow.start_run(run_name=run_id):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
            except Exception as e:
                print(f"MLflow logging warning: {e}")
                
        return run_data
        
    def get_runs(self):
        """Retrieve all recorded experiment runs."""
        try:
            with open(self.log_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
