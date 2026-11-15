import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from training.checkpoints import final_checkpoint_path, stage_checkpoint_path
from training.curriculum_report import StageSummary
from training.episode_runner import EpisodeResult
from training.evaluation import EvaluationSummary
from training.factories import build_agent_spec, build_learner_spec, build_training_spec, default_run_name, run_directory
from training.metrics_log import (
    EPISODE_FIELDS,
    EpisodeCsvLogger,
    EpisodeRecord,
    EvaluationRecord,
    evaluation_row,
    format_reasons,
    write_curriculum_matrix,
)
from training.reporting import format_matrix_table, format_stage_end
from utils import config


def read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.reader(handle))


class CheckpointPathTests(unittest.TestCase):
    def test_stage_checkpoint_path_format(self) -> None:
        self.assertEqual(stage_checkpoint_path("runs/x", 2, "best"), str(Path("runs/x/models/stage_2_best.weights.h5")))

    def test_final_checkpoint_path(self) -> None:
        self.assertEqual(final_checkpoint_path("runs/x"), str(Path("runs/x/models/final.weights.h5")))

    def test_unknown_tag_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stage_checkpoint_path("runs/x", 1, "mejor")


class MetricsLogTests(unittest.TestCase):
    def test_episode_logger_writes_header_once_and_rows(self) -> None:
        result = EpisodeResult(1, 0.5, 1, 12, "win", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics" / "episodes.csv"
            logger = EpisodeCsvLogger(str(path))
            logger.log(EpisodeRecord(1, 1, False, 1.0, 1.0, result))
            logger.log(EpisodeRecord(1, 2, True, 0.995, 1.0, result))
            rows = read_rows(path)
        self.assertEqual(rows[0], list(EPISODE_FIELDS))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], ["1", "1", "1", "0", "0.5000", "1", "12", "win", "1.0000", "", "1.0000"])
        self.assertEqual(rows[2][3], "1")

    def test_evaluation_row_formats_reasons(self) -> None:
        summary = EvaluationSummary(2, 0.25, 1.5, 30.0, {"wall": 3, "win": 1})
        row = evaluation_row(EvaluationRecord(1, 50, "periodica", summary))
        self.assertEqual(row["reasons"], "wall:3;win:1")
        self.assertEqual(row["win_rate"], "0.2500")

    def test_format_reasons_empty(self) -> None:
        self.assertEqual(format_reasons({}), "")

    def test_curriculum_matrix_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            write_curriculum_matrix(str(path), ((1.0, 0.25),), (1, 2))
            rows = read_rows(path)
        self.assertEqual(rows, [["stage_terminada", "nivel_1", "nivel_2"], ["1", "1.0000", "0.2500"]])


class ReportingFormatTests(unittest.TestCase):
    def test_stage_end_when_passed(self) -> None:
        self.assertEqual(format_stage_end(StageSummary(1, "l.json", True, 120, 0.9)), "¡Etapa 1 superada en 120 episodios!")

    def test_stage_end_without_evaluation(self) -> None:
        self.assertIn("sin evaluar", format_stage_end(StageSummary(2, "l.json", False, 10, None)))

    def test_matrix_table(self) -> None:
        table = format_matrix_table(((1.0, 0.5),), (1, 2))
        self.assertEqual(table.splitlines()[1].split(), ["1", "1.00", "0.50"])


class FactoryTests(unittest.TestCase):
    def test_agent_spec_comes_from_config(self) -> None:
        spec = build_agent_spec()
        self.assertEqual(spec.hidden_units, config.HIDDEN_UNITS)
        self.assertEqual(spec.target_rule, config.TARGET_RULE)
        self.assertEqual(spec.use_xla, config.USE_XLA)

    def test_learner_and_training_spec_come_from_config(self) -> None:
        self.assertEqual(build_learner_spec().train_every, config.TRAIN_EVERY)
        spec = build_training_spec("runs/prueba", 5)
        self.assertEqual((spec.run_dir, spec.seed, spec.level_paths), ("runs/prueba", 5, config.LEVEL_PATHS))

    def test_run_name_and_directory(self) -> None:
        self.assertEqual(default_run_name(datetime(2026, 1, 2, 3, 4, 5)), "20260102-030405")
        self.assertEqual(run_directory("abc"), str(Path(config.RUNS_DIR) / "abc"))


if __name__ == "__main__":
    unittest.main()
