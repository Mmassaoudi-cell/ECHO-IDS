"""Unified ECHO-IDS architecture.

This file collects the proposed method's core model components in one place:

1. Variance-gated temporal convolutional intrusion encoder.
2. Protocol-role graph attention and gated feature fusion.
3. Binary and multiclass IDS heads with temperature calibration.
4. Structured evidence extraction from IDS inputs and predictions.
5. Evidence-Grounded Semantic Inconsistency Index (EG-SINdex).
6. Validation-only score calibration and joint IDS/report verification.

Training loops, dataset-specific preprocessing, baselines, and experiment
orchestration remain separate because they are evaluation infrastructure rather
than model architecture.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F


class TextEncoder(Protocol):
    """Minimal interface implemented by SentenceTransformer encoders."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ECHOConfig:
    input_features: int
    graph_features: int
    attack_classes: int
    hidden_dim: int = 128
    dropout: float = 0.2
    variance_strength: float = 0.5
    attention_heads: int = 4
    use_variance_gate: bool = True
    use_graph: bool = True


@dataclass(frozen=True)
class EGSINdexWeights:
    semantic: float = 0.20
    mismatch: float = 0.35
    unsupported: float = 0.35
    coherence: float = 0.10

    def normalized(self) -> "EGSINdexWeights":
        values = np.asarray(
            [self.semantic, self.mismatch, self.unsupported, self.coherence],
            dtype=float,
        )
        if np.any(values < 0) or values.sum() <= 0:
            raise ValueError("EG-SINdex weights must be nonnegative with positive sum")
        values /= values.sum()
        return EGSINdexWeights(*values.tolist())


@dataclass
class IDSOutput:
    binary_logits: torch.Tensor
    multiclass_logits: torch.Tensor
    evidence_vector: torch.Tensor
    fusion_gate: torch.Tensor


@dataclass
class HallucinationOutput:
    raw_score: float
    probability: float
    label: int
    semantic_inconsistency: float
    evidence_mismatch: float
    unsupported_claim_penalty: float
    intra_cluster_coherence: float
    cluster_count: int
    contradicted_fields: list[str]


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.net(inputs))


class VarianceGatedTemporalEncoder(nn.Module):
    """TCN encoder whose time steps are rescaled by local feature variance."""

    def __init__(self, config: ECHOConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.input_features, config.hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(config.hidden_dim, dilation=1, dropout=config.dropout),
            ResidualTCNBlock(config.hidden_dim, dilation=2, dropout=config.dropout),
            ResidualTCNBlock(config.hidden_dim, dilation=4, dropout=config.dropout),
        )
        self.pool_gate = nn.Linear(config.hidden_dim, 1)

    def variance_gate(self, sequence: torch.Tensor) -> torch.Tensor:
        if not self.config.use_variance_gate:
            return sequence
        local_variance = sequence.var(dim=-1, unbiased=False, keepdim=True)
        reference = local_variance.mean(dim=1, keepdim=True).clamp_min(1e-6)
        gate = 1.0 + self.config.variance_strength * (
            local_variance / reference - 1.0
        )
        return sequence * gate.clamp(0.25, 4.0)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, temporal window, traffic features].
        gated = self.variance_gate(sequence)
        encoded = self.blocks(self.input_projection(gated.transpose(1, 2)))
        encoded = encoded.transpose(1, 2)
        attention = torch.softmax(self.pool_gate(encoded), dim=1)
        return torch.sum(attention * encoded, dim=1)


class ProtocolRoleGraphEncoder(nn.Module):
    """Lightweight self-attention over protocol/source/destination evidence nodes."""

    def __init__(self, config: ECHOConfig):
        super().__init__()
        self.project = nn.Linear(config.graph_features, config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, graph_nodes: torch.Tensor) -> torch.Tensor:
        nodes = self.project(graph_nodes)
        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        nodes = self.norm1(nodes + attended)
        nodes = self.norm2(nodes + self.feed_forward(nodes))
        return nodes.mean(dim=1)


class EchoIDS(nn.Module):
    """Proposed intrusion module with binary and attack-class outputs."""

    def __init__(self, config: ECHOConfig):
        super().__init__()
        self.config = config
        self.temporal_encoder = VarianceGatedTemporalEncoder(config)
        self.graph_encoder = ProtocolRoleGraphEncoder(config)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.binary_head = nn.Linear(config.hidden_dim, 2)
        self.multiclass_head = nn.Linear(
            config.hidden_dim, config.attack_classes
        )

    def encode(
        self,
        sequence: torch.Tensor,
        graph_nodes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal_encoder(sequence)
        if self.config.use_graph:
            if graph_nodes is None:
                raise ValueError("graph_nodes are required when use_graph=True")
            graph = self.graph_encoder(graph_nodes)
            gate = self.fusion_gate(torch.cat([temporal, graph], dim=-1))
            fused = gate * temporal + (1.0 - gate) * graph
        else:
            gate = torch.ones_like(temporal)
            fused = temporal
        return self.evidence_projection(fused), gate

    def forward(
        self,
        sequence: torch.Tensor,
        graph_nodes: torch.Tensor | None = None,
    ) -> IDSOutput:
        evidence_vector, gate = self.encode(sequence, graph_nodes)
        return IDSOutput(
            binary_logits=self.binary_head(evidence_vector),
            multiclass_logits=self.multiclass_head(evidence_vector),
            evidence_vector=evidence_vector,
            fusion_gate=gate,
        )


class TemperatureScaler:
    """Fits one positive temperature using validation logits only."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        def objective(log_temperature: float) -> float:
            temperature = math.exp(float(log_temperature))
            return float(
                F.cross_entropy(logits_tensor / temperature, labels_tensor)
            )

        optimum = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        self.temperature = float(math.exp(optimum.x))
        return self

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        exponent = np.exp(scaled)
        return exponent / exponent.sum(axis=1, keepdims=True)


class ValidationStackedIDS:
    """Validation-trained fusion of ECHO and an external tabular classifier."""

    def __init__(self, seed: int = 17):
        self.seed = seed
        self.stacker = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        self.threshold = 0.5

    @staticmethod
    def _features(
        echo_probabilities: np.ndarray, tabular_probabilities: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([echo_probabilities, tabular_probabilities], axis=1)

    def fit(
        self,
        echo_validation: np.ndarray,
        tabular_validation: np.ndarray,
        labels: np.ndarray,
    ) -> "ValidationStackedIDS":
        features = self._features(echo_validation, tabular_validation)
        self.stacker.fit(features, labels)
        probabilities = self.stacker.predict_proba(features)[:, 1]
        self.threshold = choose_threshold(labels, probabilities)
        return self

    def predict_proba(
        self,
        echo_probabilities: np.ndarray,
        tabular_probabilities: np.ndarray,
    ) -> np.ndarray:
        return self.stacker.predict_proba(
            self._features(echo_probabilities, tabular_probabilities)
        )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Validation macro-F1 threshold with a small false-positive penalty."""
    best_score, best_threshold = -np.inf, 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, _, _ = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        false_positive_rate = fp / max(tn + fp, 1)
        score = (
            f1_score(labels, predictions, average="macro", zero_division=0)
            - 0.05 * false_positive_rate
        )
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


class EvidenceBuilder:
    """Converts calibrated IDS outputs and input views into an evidence object."""

    def __init__(self, feature_names: Sequence[str], attack_labels: dict[int, str]):
        self.feature_names = list(feature_names)
        self.attack_labels = attack_labels

    def _categorical_value(
        self, values: np.ndarray, prefix: str, default: str = "unknown"
    ) -> str:
        candidates = [
            (name.removeprefix(prefix), float(values[index]))
            for index, name in enumerate(self.feature_names)
            if name.startswith(prefix)
        ]
        return max(candidates, key=lambda item: item[1])[0] if candidates else default

    def _port_group(self, values: np.ndarray) -> str:
        indices = [
            index
            for index, name in enumerate(self.feature_names)
            if any(
                token in name.lower()
                for token in ("dstport", "resp_p", "srcport", "orig_p")
            )
        ]
        if not indices:
            return "unspecified"
        magnitude = float(np.max(np.abs(values[indices])))
        if magnitude < 0.5:
            return "common-service"
        if magnitude < 2.0:
            return "registered"
        return "high-or-anomalous"

    @staticmethod
    def summarize(evidence: dict[str, Any]) -> str:
        leading = ", ".join(
            item["name"] for item in evidence["top_features"][:3]
        )
        return (
            f"Dataset {evidence['dataset']}; window {evidence['window_id']}; "
            f"IDS label {evidence['predicted_label']} "
            f"({evidence['attack_type']}) with confidence "
            f"{evidence['confidence']:.3f}; protocol {evidence['protocol']}; "
            f"service {evidence['service']}; port group "
            f"{evidence['port_group']}; source role "
            f"{evidence['source_role']}; destination role "
            f"{evidence['destination_role']}; severity "
            f"{evidence['severity_score']}/5; leading evidence {leading}; "
            f"recommended action: {evidence['recommended_action']}."
        )

    def build(
        self,
        *,
        dataset: str,
        sample_id: int,
        sequence: np.ndarray,
        graph_nodes: np.ndarray,
        binary_probabilities: np.ndarray,
        multiclass_probabilities: np.ndarray,
        source_rows: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        values = np.asarray(sequence).mean(axis=0)
        ranking = np.argsort(np.abs(values))[::-1][:5]
        attack_index = int(np.argmax(multiclass_probabilities))
        attack_type = self.attack_labels[attack_index]
        attack_probability = float(binary_probabilities[1])
        is_attack = attack_probability >= 0.5
        confidence = (
            attack_probability if is_attack else float(binary_probabilities[0])
        )
        graph_mean = np.asarray(graph_nodes).mean(axis=0)
        source_role = (
            "internal client" if graph_mean[0] >= 0 else "external initiator"
        )
        destination_index = min(1, len(graph_mean) - 1)
        destination_role = (
            "service endpoint"
            if graph_mean[destination_index] >= 0
            else "field device"
        )
        severity = 1 if not is_attack else int(np.clip(np.ceil(1 + 4 * confidence), 2, 5))
        action = (
            "isolate the implicated flow and validate the destination service"
            if is_attack
            else "continue monitoring"
        )
        evidence: dict[str, Any] = {
            "dataset": dataset,
            "sample_id": int(sample_id),
            "timestamp": None,
            "window_id": int(sample_id),
            "source_rows": list(source_rows or []),
            "predicted_label": "attack" if is_attack else "normal",
            "attack_type": attack_type,
            "confidence": confidence,
            "top_features": [
                {
                    "name": self.feature_names[index],
                    "standardized_magnitude": float(values[index]),
                }
                for index in ranking
            ],
            "source_role": source_role,
            "destination_role": destination_role,
            "protocol": self._categorical_value(values, "categorical__proto_"),
            "service": self._categorical_value(values, "categorical__service_"),
            "port_group": self._port_group(values),
            "packet_statistics": {
                "mean_abs_standardized": float(np.mean(np.abs(values))),
                "max_abs_standardized": float(np.max(np.abs(values))),
            },
            "traffic_rate": float(np.linalg.norm(values) / np.sqrt(len(values))),
            "temporal_variance": float(np.mean(np.var(sequence, axis=0))),
            "severity_score": severity,
            "recommended_action": action,
        }
        evidence["text_summary"] = self.summarize(evidence)
        return evidence


def _cluster_embeddings(
    embeddings: np.ndarray,
    cosine_threshold: float,
    algorithm: str,
    linkage: str,
) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    distance_threshold = max(1e-3, 1.0 - cosine_threshold)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
            distance_threshold=distance_threshold,
        ).fit_predict(embeddings)
    if algorithm == "dbscan":
        return DBSCAN(
            eps=distance_threshold, min_samples=2, metric="cosine"
        ).fit_predict(embeddings)
    if algorithm == "optics":
        return OPTICS(
            max_eps=distance_threshold, min_samples=2, metric="cosine"
        ).fit_predict(embeddings)
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def semantic_consistency(
    embeddings: np.ndarray,
    cosine_threshold: float = 0.85,
    algorithm: str = "agglomerative",
    linkage: str = "average",
) -> tuple[float, float, np.ndarray]:
    labels = _cluster_embeddings(
        embeddings, cosine_threshold, algorithm, linkage
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(
        -np.sum(probabilities * np.log(probabilities + 1e-12))
        / max(math.log(len(embeddings)), 1e-12)
    )
    similarities = 1.0 - squareform(pdist(embeddings, metric="cosine"))
    cluster_coherence = []
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            cluster_coherence.append(1.0)
            continue
        block = similarities[np.ix_(indices, indices)]
        numerator = block.sum() - len(indices)
        denominator = len(indices) * (len(indices) - 1)
        cluster_coherence.append(float(numerator / denominator))
    coherence = float(np.average(cluster_coherence, weights=counts))
    return entropy, coherence, labels


def unsupported_claims(
    report: str, evidence: dict[str, Any]
) -> tuple[float, list[str]]:
    """Returns a rule-based penalty and the contradicted/invented fields."""
    normalized = report.casefold()
    contradictions: list[str] = []
    exact_fields = {
        "attack_type": evidence["attack_type"],
        "protocol": evidence["protocol"],
        "source_role": evidence["source_role"],
        "destination_role": evidence["destination_role"],
        "port_group": evidence["port_group"],
    }
    for field, value in exact_fields.items():
        if str(value).casefold() not in normalized:
            contradictions.append(field)

    severity = re.findall(
        r"(?:severity was|severity)\D{0,15}([1-5])", normalized
    )
    if severity and int(severity[0]) != int(evidence["severity_score"]):
        contradictions.append("severity_score")

    unsupported_patterns = {
        "physical_consequence": r"emergency shutdown|unsafe pressure|physical damage",
        "timeline": r"exactly \d+ minutes|began \d+",
        "causal_explanation": r"definitively caused|stolen administrator credential",
        "unsupported_mitigation": r"reflash every|disable the plant safety",
        "invented_port": r"port was \d+|destination port was \d+",
    }
    for field, pattern in unsupported_patterns.items():
        if re.search(pattern, normalized):
            contradictions.append(field)
    unique = sorted(set(contradictions))
    return min(1.0, len(unique) / 3.0), unique


class EGSINdexVerifier:
    """Evidence-conditioned multi-response hallucination detector."""

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        weights: EGSINdexWeights | None = None,
        cosine_threshold: float = 0.85,
        algorithm: str = "agglomerative",
        linkage: str = "average",
        seed: int = 17,
    ):
        self.encoder = encoder
        self.weights = (weights or EGSINdexWeights()).normalized()
        self.cosine_threshold = cosine_threshold
        self.algorithm = algorithm
        self.linkage = linkage
        self.calibrator = LogisticRegression(random_state=seed)
        self.decision_threshold = 0.5
        self._is_calibrated = False

    def components(
        self,
        evidence: dict[str, Any],
        report: str,
        responses: Sequence[str],
    ) -> dict[str, Any]:
        if not responses:
            raise ValueError("At least one sampled report is required")
        evidence_summary = evidence["text_summary"]
        conditioned = [
            f"{evidence_summary} [SEP] {response}" for response in responses
        ]
        conditioned_embeddings = np.asarray(
            self.encoder.encode(
                conditioned,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        pair_embeddings = np.asarray(
            self.encoder.encode(
                [evidence_summary, report],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        mismatch = float(
            np.clip(
                1.0 - np.dot(pair_embeddings[0], pair_embeddings[1]),
                0.0,
                1.0,
            )
        )
        semantic, coherence, labels = semantic_consistency(
            conditioned_embeddings,
            self.cosine_threshold,
            self.algorithm,
            self.linkage,
        )
        unsupported, contradicted_fields = unsupported_claims(report, evidence)
        raw_score = (
            self.weights.semantic * semantic
            + self.weights.mismatch * mismatch
            + self.weights.unsupported * unsupported
            + self.weights.coherence * (1.0 - coherence)
        )
        return {
            "raw_score": float(raw_score),
            "semantic_inconsistency": semantic,
            "evidence_mismatch": mismatch,
            "unsupported_claim_penalty": unsupported,
            "intra_cluster_coherence": coherence,
            "cluster_count": int(np.unique(labels).size),
            "contradicted_fields": contradicted_fields,
        }

    def fit_calibration(
        self,
        validation_examples: Sequence[
            tuple[dict[str, Any], str, Sequence[str], int]
        ],
    ) -> "EGSINdexVerifier":
        """Fits probability calibration and threshold on validation examples."""
        scores = np.asarray(
            [
                self.components(evidence, report, responses)["raw_score"]
                for evidence, report, responses, _ in validation_examples
            ],
            dtype=float,
        ).reshape(-1, 1)
        labels = np.asarray(
            [label for _, _, _, label in validation_examples], dtype=int
        )
        self.calibrator.fit(scores, labels)
        validation_probabilities = self.calibrator.predict_proba(scores)[:, 1]
        self.decision_threshold = choose_threshold(
            labels, validation_probabilities
        )
        self._is_calibrated = True
        return self

    def verify(
        self,
        evidence: dict[str, Any],
        report: str,
        responses: Sequence[str],
    ) -> HallucinationOutput:
        components = self.components(evidence, report, responses)
        raw_score = components["raw_score"]
        if self._is_calibrated:
            probability = float(
                self.calibrator.predict_proba([[raw_score]])[0, 1]
            )
        else:
            probability = float(np.clip(raw_score, 0.0, 1.0))
        return HallucinationOutput(
            probability=probability,
            label=int(probability >= self.decision_threshold),
            **components,
        )


class ECHOSystem(nn.Module):
    """Single interface for intrusion inference, evidence, and report checking."""

    def __init__(
        self,
        ids_model: EchoIDS,
        evidence_builder: EvidenceBuilder,
        verifier: EGSINdexVerifier | None = None,
        binary_temperature: TemperatureScaler | None = None,
        multiclass_temperature: TemperatureScaler | None = None,
    ):
        super().__init__()
        self.ids_model = ids_model
        self.evidence_builder = evidence_builder
        self.verifier = verifier
        self.binary_temperature = binary_temperature or TemperatureScaler()
        self.multiclass_temperature = multiclass_temperature or TemperatureScaler()

    @torch.inference_mode()
    def detect_and_build_evidence(
        self,
        *,
        dataset: str,
        sample_id: int,
        sequence: np.ndarray,
        graph_nodes: np.ndarray,
        source_rows: Sequence[int] | None = None,
    ) -> tuple[dict[str, Any], IDSOutput]:
        device = next(self.ids_model.parameters()).device
        sequence_tensor = torch.as_tensor(
            sequence, dtype=torch.float32, device=device
        ).unsqueeze(0)
        graph_tensor = torch.as_tensor(
            graph_nodes, dtype=torch.float32, device=device
        ).unsqueeze(0)
        output = self.ids_model(sequence_tensor, graph_tensor)
        binary_probability = self.binary_temperature.probabilities(
            output.binary_logits.detach().cpu().numpy()
        )[0]
        multiclass_probability = self.multiclass_temperature.probabilities(
            output.multiclass_logits.detach().cpu().numpy()
        )[0]
        evidence = self.evidence_builder.build(
            dataset=dataset,
            sample_id=sample_id,
            sequence=sequence,
            graph_nodes=graph_nodes,
            binary_probabilities=binary_probability,
            multiclass_probabilities=multiclass_probability,
            source_rows=source_rows,
        )
        evidence["evidence_vector"] = (
            output.evidence_vector[0].detach().cpu().tolist()
        )
        evidence["fusion_gate_mean"] = float(
            output.fusion_gate[0].mean().detach().cpu()
        )
        return evidence, output

    def verify_report(
        self,
        evidence: dict[str, Any],
        report: str,
        responses: Sequence[str],
    ) -> HallucinationOutput:
        if self.verifier is None:
            raise RuntimeError("No EG-SINdex verifier was configured")
        return self.verifier.verify(evidence, report, responses)

    def joint_inference(
        self,
        *,
        dataset: str,
        sample_id: int,
        sequence: np.ndarray,
        graph_nodes: np.ndarray,
        report: str,
        responses: Sequence[str],
        source_rows: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        evidence, _ = self.detect_and_build_evidence(
            dataset=dataset,
            sample_id=sample_id,
            sequence=sequence,
            graph_nodes=graph_nodes,
            source_rows=source_rows,
        )
        verification = self.verify_report(evidence, report, responses)
        return {
            "evidence": evidence,
            "hallucination": asdict(verification),
        }


def load_sentence_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> TextEncoder:
    """Loads the default semantic encoder only when requested."""
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


__all__ = [
    "ECHOConfig",
    "EGSINdexWeights",
    "IDSOutput",
    "HallucinationOutput",
    "EchoIDS",
    "TemperatureScaler",
    "ValidationStackedIDS",
    "EvidenceBuilder",
    "EGSINdexVerifier",
    "ECHOSystem",
    "choose_threshold",
    "semantic_consistency",
    "unsupported_claims",
    "load_sentence_encoder",
]
