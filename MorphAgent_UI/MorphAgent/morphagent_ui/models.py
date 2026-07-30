"""Dependency-light state, validation, and CLI construction for the UI.

This module deliberately does not import MorphAgent's scientific runtime.  The
desktop shell can therefore explain missing dependencies and credentials before
the heavyweight pipeline is started in its own process.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
VLM_NATIVE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MASK_DIR_MARKERS = ("segmentation", "segment", "mask")


def _environment_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _environment_ratio(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if 0.0 <= value <= 1.0 else default


class Severity(str, Enum):
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    recovery: str = ""


@dataclass(frozen=True)
class SampleSummary:
    sample_id: str
    primary_images: int
    vlm_source_images: int
    vlm_native_images: int
    segmentation_masks: int
    vlm_source: str = "none"

    @property
    def has_usable_image(self) -> bool:
        return self.primary_images > 0 or self.vlm_source_images > 0


@dataclass(frozen=True)
class DatasetSummary:
    requested_root: str
    resolved_root: str
    samples: tuple[SampleSummary, ...] = ()
    used_dataset_child: bool = False

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def empty_samples(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples if not sample.has_usable_image)

    @property
    def primary_image_count(self) -> int:
        return sum(sample.primary_images for sample in self.samples)

    @property
    def vlm_source_count(self) -> int:
        return sum(sample.vlm_source_images for sample in self.samples)

    @property
    def vlm_native_count(self) -> int:
        return sum(sample.vlm_native_images for sample in self.samples)

    @property
    def mask_count(self) -> int:
        return sum(sample.segmentation_masks for sample in self.samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_root": self.requested_root,
            "resolved_root": self.resolved_root,
            "used_dataset_child": self.used_dataset_child,
            "sample_count": self.sample_count,
            "empty_samples": list(self.empty_samples),
            "primary_image_count": self.primary_image_count,
            "vlm_source_count": self.vlm_source_count,
            "vlm_native_count": self.vlm_native_count,
            "mask_count": self.mask_count,
            "samples": [asdict(sample) for sample in self.samples],
        }


class RunPreset(str, Enum):
    PILOT = "pilot"

    @property
    def title(self) -> str:
        return "Demo · both routes"

    @property
    def description(self) -> str:
        return "Demo scale · 1 round × 5 candidates · target 5"


@dataclass
class RunConfig:
    data_root: str = ""
    query: str = ""
    results_dir: str = ""
    description_path: str = ""
    metadata_path: str = ""
    method: str = "both"
    features_per_iteration: int = field(default_factory=lambda: _environment_int("FEATURES_PER_ITERATION", 5))
    target_feature_count: int = field(default_factory=lambda: _environment_int("TARGET_FEATURE_COUNT", 5))
    num_rounds: int = field(default_factory=lambda: _environment_int("NUM_ROUNDS", 1))
    enable_expert_knowledge: bool = True
    enable_deep_research: bool = True
    enable_rag: bool = True
    enable_background_knowledge_in_planning: bool = True
    enable_segmentation: bool = True
    segmentation_skip_if_present: bool = True
    enable_feature_analysis: bool = True
    reproduce: bool = True
    reproduce_seed: int = 42
    resume: bool = False
    temperature: float = 0.0
    code_vlm_ratio: float = field(default_factory=lambda: _environment_ratio("CODE_VLM_RATIO", 0.5))
    knowledge_dependency: float = field(default_factory=lambda: _environment_ratio("KNOWLEDGE_DEPENDENCY", 0.5))
    code_parallel_workers: int = field(default_factory=lambda: _environment_int("CODE_PARALLEL_WORKERS", 1))
    vlm_online_concurrency: int = field(default_factory=lambda: _environment_int("VLM_ONLINE_CONCURRENCY", 1))
    multigpu: bool = False
    api_provider: str = "default"
    vlm_api_provider: str = "online"
    llm_model: str = ""
    vlm_online_model: str = ""
    dataset_source: str = "custom"  # "demo" | "custom"
    pubmed_max_results: int = 10
    # Live API fields from Configure (used for preflight before .env write).
    llm_base_url: str = ""
    llm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    reuse_llm_for_vlm: bool = False
    python_executable: str = field(default_factory=lambda: sys.executable)
    repository_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[1]))

    def apply_preset(self, preset: RunPreset | str) -> None:
        RunPreset(preset)
        self.method = "both"
        self.features_per_iteration = 5
        self.target_feature_count = 5
        self.num_rounds = 1
        self.temperature = 0.0
        self.reproduce = True
        self.code_parallel_workers = 1
        self.vlm_online_concurrency = 1

    def apply_reference_demo(self) -> Path:
        """Load the demo dataset and seed its precomputed RAG cache."""

        repository = Path(self.repository_root).expanduser().resolve()
        demo = repository / "demo"
        project_root = demo / "data"
        dataset = project_root / "dataset"
        description = dataset / "dataset_index.txt"
        rag_dir = project_root / "RAG"
        precomputed = demo / "precomputed" / "rag_knowledge_summary.txt"
        required = (project_root, dataset, description, rag_dir, precomputed)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Demo dataset is incomplete: " + ", ".join(missing))

        pdf_files = sorted(rag_dir.glob("*.pdf"))
        xml_files = sorted(rag_dir.glob("*.xml"))
        if not pdf_files and not xml_files:
            raise FileNotFoundError(f"Demo RAG folder has no PDF/XML files: {rag_dir}")

        # Import only for this explicit action so routine UI preflight remains
        # dependency-light.
        from knowledge.rag import _compute_rag_folder_hash

        rag_hash = _compute_rag_folder_hash(rag_dir, pdf_files, xml_files)
        cache_dir = project_root / ".rag_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"rag_cache_{rag_hash}.txt"
        content = precomputed.read_text(encoding="utf-8")
        metadata = {
            "hash": rag_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_length": len(content),
        }
        cache_path.write_text(
            "# RAG Cache Metadata\n"
            f"# {json.dumps(metadata, ensure_ascii=False)}\n"
            "# End Metadata\n\n"
            f"{content}",
            encoding="utf-8",
        )

        self.data_root = str(project_root)
        self.description_path = str(description)
        metadata_csv = project_root / "metadata.csv"
        self.metadata_path = str(metadata_csv.resolve()) if metadata_csv.is_file() else ""
        self.results_dir = ""
        self.query = (
            "Generate unbiased morphological features that quantify Tau protein "
            "aggregation and neuronal structure in these images"
        )
        self.apply_preset(RunPreset.PILOT)
        self.dataset_source = "demo"
        self.enable_expert_knowledge = True
        self.enable_deep_research = True
        self.enable_rag = True
        self.enable_background_knowledge_in_planning = True
        self.enable_segmentation = True
        self.segmentation_skip_if_present = True
        self.enable_feature_analysis = True
        self.resume = False
        return cache_path

    @property
    def resolved_results_dir(self) -> Path | None:
        if self.results_dir.strip():
            return Path(self.results_dir).expanduser().resolve()
        if not self.data_root.strip():
            return None
        root = Path(self.data_root).expanduser().resolve()
        dataset = root / "dataset" if (root / "dataset").is_dir() else root
        return dataset.parent / "results"

    def validate(
        self,
        dataset: DatasetSummary | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> list[ValidationIssue]:
        env = os.environ if environment is None else environment
        issues: list[ValidationIssue] = []
        root = Path(self.data_root).expanduser() if self.data_root.strip() else None
        python = Path(self.python_executable).expanduser() if self.python_executable.strip() else None

        if root is None:
            issues.append(ValidationIssue(Severity.BLOCKER, "dataset_missing", "Choose a dataset folder.", "Select the project root or dataset root."))
        elif not root.exists() or not root.is_dir():
            issues.append(ValidationIssue(Severity.BLOCKER, "dataset_invalid", "The dataset folder does not exist.", "Choose an existing folder."))
        elif dataset is not None:
            if dataset.sample_count == 0:
                issues.append(ValidationIssue(Severity.BLOCKER, "samples_missing", "No sample folders were found.", "Add one non-hidden folder per sample."))
            if dataset.empty_samples:
                preview = ", ".join(dataset.empty_samples[:4])
                suffix = "…" if len(dataset.empty_samples) > 4 else ""
                issues.append(ValidationIssue(Severity.BLOCKER, "empty_samples", f"Samples without usable images: {preview}{suffix}", "Remove empty folders or add microscopy images."))
            if self.method in {"code", "both"} and dataset.primary_image_count == 0:
                issues.append(ValidationIssue(Severity.BLOCKER, "code_sources_missing", "The code route has no primary images.", "Place raw images directly inside each sample folder or use VLM-only."))
            if self.method in {"vlm", "both"} and dataset.vlm_source_count == 0:
                issues.append(ValidationIssue(Severity.BLOCKER, "vlm_sources_missing", "The VLM route has no image source.", "Add primary images or a non-mask image subfolder."))
            elif self.method in {"vlm", "both"} and dataset.vlm_native_count < dataset.vlm_source_count:
                issues.append(ValidationIssue(Severity.INFO, "vlm_preparation", "Some VLM inputs need PNG/JPEG slice preparation.", "MorphAgent prepares compatible 2D views during the first round."))
            if dataset.sample_count > 0 and dataset.sample_count < 5:
                issues.append(ValidationIssue(
                    Severity.WARNING,
                    "sample_count_low",
                    f"Only {dataset.sample_count} samples were found (recommend ≥5).",
                    "Small datasets can fail validation due to insufficient unique values. Add more samples when possible.",
                ))

        if not self.query.strip():
            issues.append(ValidationIssue(Severity.BLOCKER, "query_missing", "Describe the biological question.", "State the phenotype, object, or comparison you want to profile."))
        if python is None or not python.exists():
            issues.append(ValidationIssue(Severity.BLOCKER, "python_missing", "The selected Python interpreter was not found.", "Choose the interpreter from the MorphAgent environment."))
        if self.method not in {"code", "vlm", "both"}:
            issues.append(ValidationIssue(Severity.BLOCKER, "method_invalid", "Choose code, VLM, or both routes."))
        if self.features_per_iteration < 1 or self.target_feature_count < 1 or self.num_rounds < 1:
            issues.append(ValidationIssue(Severity.BLOCKER, "counts_invalid", "Feature counts and rounds must be positive."))
        if self.target_feature_count < self.features_per_iteration:
            issues.append(ValidationIssue(Severity.WARNING, "target_small", "The target is smaller than one planned round.", "Increase the target or lower candidates per round."))
        if not 0.0 <= self.code_vlm_ratio <= 1.0 or not 0.0 <= self.knowledge_dependency <= 1.0:
            issues.append(ValidationIssue(Severity.BLOCKER, "ratio_invalid", "Route and knowledge ratios must be between 0 and 1."))

        if self.description_path and not Path(self.description_path).expanduser().is_file():
            issues.append(ValidationIssue(Severity.BLOCKER, "description_invalid", "The dataset description file was not found."))
        if self.enable_feature_analysis and self.metadata_path.strip():
            metadata = Path(self.metadata_path).expanduser()
            if not metadata.is_file():
                issues.append(ValidationIssue(Severity.BLOCKER, "metadata_invalid", "The metadata CSV was not found."))
            elif metadata.suffix.lower() != ".csv":
                issues.append(ValidationIssue(Severity.WARNING, "metadata_format", "Metadata is expected to be a CSV file."))
        elif self.enable_feature_analysis and not self.metadata_path.strip():
            issues.append(ValidationIssue(
                Severity.INFO,
                "metadata_optional",
                "Feature validation is on without metadata — unsupervised checks will run.",
                "Optional: provide a metadata.csv with sample_id + group/label columns for paired validation.",
            ))

        config_path = Path(self.repository_root).expanduser() / "config.py"
        source_llm = _configured_key_fallback(config_path, "DEFAULT_LLM_API_KEY")
        source_vlm = _configured_key_fallback(config_path, "DEFAULT_VLM_API_KEY") or source_llm
        form_llm_key = (self.llm_api_key or "").strip()
        form_vlm_key = (self.vlm_api_key or "").strip()
        if self.reuse_llm_for_vlm:
            form_vlm_key = form_llm_key or form_vlm_key
        llm_key = form_llm_key or bool(str(env.get("LLM_API_KEY", "")).strip()) or source_llm
        vlm_key = (
            form_vlm_key
            or bool(str(env.get("VLM_API_KEY", "")).strip())
            or (form_llm_key if self.reuse_llm_for_vlm or not (self.vlm_base_url or self.vlm_online_model) else False)
            or llm_key
            or source_vlm
        )
        llm_base = (self.llm_base_url or str(env.get("LLM_BASE_URL", ""))).strip()
        llm_model = (self.llm_model or str(env.get("LLM_MODEL", ""))).strip()
        if not llm_base or not llm_model or not llm_key:
            if self.api_provider.strip().lower() == "default":
                issues.append(ValidationIssue(
                    Severity.BLOCKER,
                    "llm_key_missing",
                    "Fill Base URL, API key, and Model under Model API.",
                    "Credentials are applied automatically when you click Run — no separate Save step.",
                ))
            else:
                issues.append(ValidationIssue(Severity.WARNING, "llm_preset_unverified", f"The UI cannot verify credentials for provider preset '{self.api_provider}'.", "Confirm that the preset's key environment variable is exported."))
        if self.method in {"vlm", "both"} and self.vlm_api_provider.lower() in {"online", "api"} and not vlm_key:
            issues.append(ValidationIssue(
                Severity.BLOCKER,
                "vlm_key_missing",
                "Image scoring needs a VLM API key (or enable “Use the same connection”).",
                "Fill the VLM fields, or check “Use the same connection for image scoring”.",
            ))
        if self.method in {"vlm", "both"} and self.vlm_api_provider.lower() == "qwen":
            issues.append(ValidationIssue(Severity.WARNING, "local_vlm", "Local Qwen requires a supported CUDA GPU and extra packages.", "Confirm the local model and device in the repository configuration."))

        if self.reproduce or self.temperature <= 0.0:
            issues.append(ValidationIssue(
                Severity.INFO,
                "reproducible_run",
                "Temperature 0 · reproducible mode (fixed seed, deterministic VLM decoding).",
                "",
            ))

        if self.resume:
            if not self.results_dir.strip():
                issues.append(ValidationIssue(Severity.BLOCKER, "resume_results_missing", "Resume requires an existing results directory."))
            else:
                results = Path(self.results_dir).expanduser()
                markers = list(results.glob("round_*/round_results.json")) if results.is_dir() else []
                if not markers:
                    issues.append(ValidationIssue(Severity.BLOCKER, "resume_marker_missing", "No completed round marker was found in this run.", "Choose a run containing round_N/round_results.json."))

        if self.method in {"code", "both"}:
            issues.append(ValidationIssue(Severity.WARNING, "generated_code", "Generated feature code executes in the configured Conda environment.", "Use trusted data and review the audit/code artifacts after the run."))
        if self.enable_segmentation and dataset is not None and dataset.mask_count < dataset.sample_count:
            issues.append(ValidationIssue(
                Severity.WARNING,
                "segmentation_needed",
                "Some samples have no masks yet; Allen segmentation will run when available.",
                "Install the optional morphagent_allen environment, or add masks under each sample's segmentation/ folder.",
            ))
        if self.data_root.strip():
            issues.append(ValidationIssue(Severity.INFO, "input_writes", "Preparation may add slices/ and segmentation/ artifacts to the dataset.", "Work on a backed-up or writable dataset."))
        return issues

    def build_command(self) -> list[str]:
        repo = Path(self.repository_root).expanduser().resolve()
        command = [self.python_executable, "-u", str(repo / "main.py"), self.query.strip(), "--data-root", str(Path(self.data_root).expanduser().resolve())]
        if self.description_path.strip():
            command.extend(["--description", str(Path(self.description_path).expanduser().resolve())])
        if self.enable_feature_analysis and self.metadata_path.strip():
            command.extend(["--metadata-path", str(Path(self.metadata_path).expanduser().resolve())])
        if self.results_dir.strip():
            command.extend(["--results-dir", str(Path(self.results_dir).expanduser().resolve())])

        command.extend([
            "--method", self.method,
            "--features-per-iteration", str(self.features_per_iteration),
            "--target-feature-count", str(self.target_feature_count),
            "--num-rounds", str(self.num_rounds),
            "--temperature", str(self.temperature),
            "--code-vlm-ratio", str(self.code_vlm_ratio),
            "--knowledge-dependency", str(self.knowledge_dependency),
            "--code-parallel-workers", str(self.code_parallel_workers),
            "--vlm-online-concurrency", str(self.vlm_online_concurrency),
            "--api-provider", self.api_provider,
            "--vlm-api-provider", self.vlm_api_provider,
        ])
        if self.llm_model.strip():
            command.extend(["--llm-model", self.llm_model.strip()])
        if self.vlm_online_model.strip():
            command.extend(["--vlm-online-model", self.vlm_online_model.strip()])

        for enabled, disable_flag in (
            (self.enable_expert_knowledge, "--disable-expert-knowledge"),
            (self.enable_deep_research, "--disable-deep-research"),
            (self.enable_rag, "--disable-rag"),
            (self.enable_background_knowledge_in_planning, "--disable-background-knowledge-in-planning"),
            (self.enable_feature_analysis, "--disable-feature-analysis"),
        ):
            if not enabled:
                command.append(disable_flag)
        command.append("--enable-segmentation" if self.enable_segmentation else "--disable-segmentation")
        # Masks are always reused when present; missing masks use Allen internally.
        command.append("--segmentation-skip-if-present")

        # Demo path digests prepared knowledge folders only. Custom datasets may
        # optionally auto-generate deep research / pull PubMed literature.
        if self.dataset_source != "demo":
            if self.enable_deep_research:
                command.append("--auto-deep-research")
            if self.enable_rag:
                command.extend([
                    "--auto-literature-retrieval",
                    "--pubmed-max-results",
                    str(max(1, int(self.pubmed_max_results))),
                ])

        if self.reproduce:
            command.extend(["--reproduce", "--reproduce-seed", str(self.reproduce_seed)])
        if self.resume:
            command.append("--resume")
        if self.multigpu:
            command.append("--multigpu")
        # UI: skip VLM critic when code features run (noisy false ❌ / wasted retries).
        if self.method in {"code", "both"}:
            command.append("--disable-critic-agent")
        return command

    def pipeline_environment(self) -> dict[str, str]:
        """Extra env vars injected when the UI launches main.py."""

        deterministic = self.reproduce or self.temperature <= 0.0
        env = {
            "CODE_MAX_RETRIES": "3",
            # Always pin Allen + its isolated env for UI runs (ignore a polluted
            # parent shell that may still have SEGMENTATION_CONDA_ENV=morphagent).
            "SEGMENTATION_BACKEND": "allen",
            "SEGMENTATION_CONDA_ENV": "morphagent_allen",
            "NUM_ROUNDS": str(self.num_rounds),
            "FEATURES_PER_ITERATION": str(self.features_per_iteration),
            "TARGET_FEATURE_COUNT": str(self.target_feature_count),
            "PUBMED_MAX_RESULTS": str(self.pubmed_max_results),
            "CODE_TEMPERATURE": "0" if deterministic else str(self.temperature),
            "VLM_TEMPERATURE": "0" if deterministic else str(self.temperature),
        }
        if self.method in {"code", "both"}:
            env["ENABLE_CRITIC_AGENT"] = "false"
        return env

    def command_preview(self) -> str:
        return shlex.join(self.build_command())

    def manifest(self, dataset: DatasetSummary | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("repository_root", None)
        for secret in ("llm_api_key", "vlm_api_key"):
            payload.pop(secret, None)
        payload["python_executable"] = str(Path(self.python_executable).expanduser())
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        payload["schema_version"] = 1
        payload["command"] = self.build_command()
        payload["dataset_summary"] = dataset.as_dict() if dataset else None
        payload["secret_policy"] = "API keys are inherited from the launch environment and are not persisted."
        return payload

    def write_manifest(self, directory: str | Path, dataset: DatasetSummary | None = None) -> Path:
        destination = Path(directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "ui_run_manifest.json"
        path.write_text(json.dumps(self.manifest(dataset), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def _image_files(directory: Path, extensions: set[str] = IMAGE_EXTENSIONS) -> list[Path]:
    try:
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    except OSError:
        return []


def _configured_key_fallback(config_path: Path, assignment_name: str) -> bool:
    """Detect only whether a source-config fallback is non-empty; never return it."""
    try:
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            values[node.target.id] = node.value

    def nonempty(name: str, seen: set[str]) -> bool:
        if name in seen:
            return False
        seen.add(name)
        value = values.get(name)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return bool(value.value.strip())
        if isinstance(value, ast.Name):
            return nonempty(value.id, seen)
        if isinstance(value, ast.Call) and len(value.args) >= 2:
            fallback = value.args[1]
            if isinstance(fallback, ast.Constant) and isinstance(fallback.value, str):
                return bool(fallback.value.strip())
            if isinstance(fallback, ast.Name):
                return nonempty(fallback.id, seen)
        return False

    return nonempty(assignment_name, set())


def _is_mask_dir(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in MASK_DIR_MARKERS)


def diagnose_dataset_selection(path: str | Path | None) -> str | None:
    """Return a user-facing error if the chosen path cannot be used, else None."""

    if path is None or not str(path).strip():
        return None
    requested = Path(path).expanduser()
    if not requested.exists():
        return (
            "This path does not exist.\n\n"
            "Choose a folder that contains a `dataset/` directory:\n\n"
            "  your_folder/\n"
            "    dataset/\n"
            "      sample_a/\n"
            "        image.tif\n"
            "      sample_b/\n"
            "        image.tif"
        )
    if not requested.is_dir():
        return "Please choose a folder (not a file). The folder should contain `dataset/` with one subfolder per sample."

    summary = scan_dataset(requested)
    dataset_child = requested / "dataset"
    if summary.sample_count == 0:
        if not dataset_child.is_dir():
            return (
                "No usable samples found.\n\n"
                f"Selected: {requested}\n\n"
                "Expected layout:\n"
                "  <path you select>/\n"
                "    dataset/\n"
                "      <sample_folder>/\n"
                "        *.tif  (primary image)\n\n"
                "Select the parent folder that contains `dataset/`, not a single sample folder."
            )
        return (
            "Found `dataset/`, but it has no sample folders with images.\n\n"
            "Each sample must be a subfolder under `dataset/` containing at least one `.tif` / `.tiff` / `.png` image."
        )
    if summary.empty_samples and len(summary.empty_samples) == summary.sample_count:
        preview = ", ".join(summary.empty_samples[:4])
        suffix = "…" if len(summary.empty_samples) > 4 else ""
        return (
            "Sample folders were found, but none contain usable images.\n\n"
            f"Empty samples: {preview}{suffix}\n\n"
            "Put a microscopy image (e.g. `image.tif`) directly inside each sample folder."
        )
    return None


def scan_dataset(path: str | Path) -> DatasetSummary:
    requested = Path(path).expanduser().resolve()
    candidate = requested / "dataset"
    resolved = candidate if candidate.is_dir() else requested
    samples: list[SampleSummary] = []
    if not resolved.is_dir():
        return DatasetSummary(str(requested), str(resolved), (), candidate.is_dir())

    try:
        sample_dirs = sorted(item for item in resolved.iterdir() if item.is_dir() and not item.name.startswith("."))
    except OSError:
        sample_dirs = []
    for sample_dir in sample_dirs:
        primary = _image_files(sample_dir)
        child_sources: list[tuple[str, list[Path]]] = []
        try:
            child_dirs = sorted(item for item in sample_dir.iterdir() if item.is_dir() and not item.name.startswith("."))
        except OSError:
            child_dirs = []
        for child in child_dirs:
            images = _image_files(child)
            if images:
                child_sources.append((child.name, images))

        non_mask = [entry for entry in child_sources if not _is_mask_dir(entry[0])]
        if non_mask:
            source_name, source_images = sorted(non_mask, key=lambda item: (-len(item[1]), item[0]))[0]
            source_kind = f"folder:{source_name}"
        elif primary:
            source_images = primary
            source_kind = "primary"
        elif child_sources:
            source_name, source_images = sorted(child_sources, key=lambda item: (-len(item[1]), item[0]))[0]
            source_kind = f"fallback:{source_name}"
        else:
            source_images = []
            source_kind = "none"
        masks = _image_files(sample_dir / "segmentation")
        samples.append(SampleSummary(
            sample_id=sample_dir.name,
            primary_images=len(primary),
            vlm_source_images=len(source_images),
            vlm_native_images=sum(path.suffix.lower() in VLM_NATIVE_EXTENSIONS for path in source_images),
            segmentation_masks=len(masks),
            vlm_source=source_kind,
        ))
    return DatasetSummary(str(requested), str(resolved), tuple(samples), resolved == candidate)


def discover_feature_names(csv_path: str | Path) -> list[str]:
    """Read only the header of a feature matrix, excluding the sample identifier."""
    path = Path(csv_path)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error):
        return []
    return [name for name in header if name and name != "sample_id"]


def find_completed_rounds(results_dir: str | Path) -> list[int]:
    completed: list[int] = []
    root = Path(results_dir)
    if not root.is_dir():
        return completed
    for marker in root.glob("round_*/round_results.json"):
        try:
            completed.append(int(marker.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(completed))


@dataclass(frozen=True)
class FeatureCard:
    feature_id: str
    name: str
    method: str = "unknown"
    category: str = "other"
    description: str = ""
    status: str = "planned"
    round_number: int = 0
    validation_score: float | None = None
    reason_codes: tuple[str, ...] = ()
    method_rationale: str = ""
    expected_visual_signature: str = ""
    required_channels: str = ""
    required_masks: str = ""
    candidate_operators: str = ""
    summary_statistics: str = ""
    source_paths: Mapping[str, str] = field(default_factory=dict)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def load_feature_cards(results_dir: str | Path) -> list[FeatureCard]:
    """Load feature cards from registry, plans, or matrix headers in that order."""
    root = Path(results_dir)
    planned: dict[str, dict[str, Any]] = {}
    for plan_path in sorted(root.glob("round_*/feature_plan.json")):
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            round_number = int(plan_path.parent.name.split("_", 1)[1])
        except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
            continue
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            continue
        for index, raw in enumerate(features):
            if not isinstance(raw, dict):
                continue
            name = _as_text(raw.get("name")).strip()
            if not name:
                continue
            entry = dict(raw)
            entry["round_number"] = round_number
            entry["feature_id"] = _as_text(raw.get("feature_id")) or f"round_{round_number}:{index}:{name}"
            planned[name] = entry

    registry_entries: list[dict[str, Any]] = []
    registry_path = root / "feature_registry.json"
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            entries = registry.get("entries", []) if isinstance(registry, dict) else []
            registry_entries = [entry for entry in entries if isinstance(entry, dict)]
        except (OSError, UnicodeError, json.JSONDecodeError):
            registry_entries = []

    cards: list[FeatureCard] = []
    consumed: set[str] = set()
    for entry in registry_entries:
        name = _as_text(entry.get("actual_column_name") or entry.get("name")).strip()
        if not name:
            continue
        plan = planned.get(name, planned.get(_as_text(entry.get("name")), {}))
        history = entry.get("decision_history") or []
        latest = history[-1] if isinstance(history, list) and history and isinstance(history[-1], dict) else {}
        raw_score = latest.get("validation_score")
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None
        cards.append(FeatureCard(
            feature_id=_as_text(entry.get("feature_id")) or name,
            name=name,
            method=_as_text(entry.get("method") or plan.get("method")) or "unknown",
            category=_as_text(entry.get("category") or plan.get("category")) or "other",
            description=_as_text(entry.get("description") or plan.get("description")),
            status=_as_text(entry.get("current_status") or latest.get("status")) or "validated",
            round_number=int(entry.get("latest_round") or plan.get("round_number") or 0),
            validation_score=score,
            reason_codes=tuple(_as_text(item) for item in latest.get("reason_codes", []) if item),
            method_rationale=_as_text(plan.get("method_rationale")),
            expected_visual_signature=_as_text(plan.get("expected_visual_signature") or plan.get("visual_signature")),
            required_channels=_as_text(plan.get("required_channels") or plan.get("channels")),
            required_masks=_as_text(plan.get("required_masks") or plan.get("segmentation_prompt")),
            candidate_operators=_as_text(plan.get("candidate_operators") or plan.get("operators")),
            summary_statistics=_as_text(plan.get("summary_statistics") or plan.get("statistics")),
            source_paths=entry.get("source_paths", {}) if isinstance(entry.get("source_paths"), dict) else {},
        ))
        consumed.add(name)

    for name, plan in planned.items():
        if name in consumed:
            continue
        cards.append(FeatureCard(
            feature_id=_as_text(plan.get("feature_id")) or name,
            name=name,
            method=_as_text(plan.get("method")) or "unknown",
            category=_as_text(plan.get("category")) or "other",
            description=_as_text(plan.get("description")),
            status="planned",
            round_number=int(plan.get("round_number") or 0),
            method_rationale=_as_text(plan.get("method_rationale")),
            expected_visual_signature=_as_text(plan.get("expected_visual_signature") or plan.get("visual_signature")),
            required_channels=_as_text(plan.get("required_channels") or plan.get("channels")),
            required_masks=_as_text(plan.get("required_masks") or plan.get("segmentation_prompt")),
            candidate_operators=_as_text(plan.get("candidate_operators") or plan.get("operators")),
            summary_statistics=_as_text(plan.get("summary_statistics") or plan.get("statistics")),
        ))

    if not cards:
        csv_path = root / "retained_features.csv"
        if not csv_path.is_file():
            csv_path = root / "features.csv"
        cards = [FeatureCard(name=name, feature_id=name, status="matrix column") for name in discover_feature_names(csv_path)]
    return sorted(cards, key=lambda card: (card.round_number, card.category, card.name))


def list_result_artifacts(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    if not root.is_dir():
        return []
    preferred: list[Path] = []
    for pattern in (
        "features.csv",
        "retained_features.csv",
        "feature_registry.json",
        "segmentation_summary.json",
        "first_sample_visualization/**/*",
        "round_*/feature_plan.json",
        "round_*/round_results.json",
        "round_*/validation_*.csv",
        "round_*/validation_*.json",
        "round_*/merged_feature_code.py",
        "round_*/merged_features/*.py",
        "round_*/merged_features/*.txt",
        "round_*/*.png",
        "round_*/*.pdf",
        "*_knowledge_summary.txt",
        "ui_console.log",
        "ui_run_manifest.json",
    ):
        preferred.extend(sorted(root.glob(pattern)))
    preferred.extend(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in preferred:
        if path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
