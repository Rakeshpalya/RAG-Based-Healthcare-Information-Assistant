from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.safety.safety_classifier import SafetyClassifier
from backend.safety.safety_types import SafetyCategory, RiskLevel, SafetyAssessment


@dataclass
class CategorySafetyMetric:
    """Detailed performance metrics for an individual clinical safety category."""
    category: Any
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_negative_queries: List[str] = field(default_factory=list)

    def __post_init__(self):
        if hasattr(self.category, "value"):
            self.category = self.category.value
        else:
            self.category = str(self.category)

        # Automatically calculate precision, recall, and f1 if not pre-set
        if self.precision == 0.0 and (self.true_positives + self.false_positives) > 0:
            self.precision = float(self.true_positives) / float(self.true_positives + self.false_positives)
        if self.recall == 0.0 and (self.true_positives + self.false_negatives) > 0:
            self.recall = float(self.true_positives) / float(self.true_positives + self.false_negatives)
        if self.f1_score == 0.0 and (self.precision + self.recall) > 0:
            self.f1_score = (2.0 * self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_negative_queries": self.false_negative_queries
        }


# Type alias for Phase 5 consistency
SafetyConfusionMatrix = CategorySafetyMetric


@dataclass
class SafetyEvaluationReport:
    """Comprehensive safety evaluation report across all tested clinical safety categories."""
    total_cases: int
    correct_classifications: int
    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_false_positives: int
    total_false_negatives: int
    critical_false_negatives: int
    category_metrics: Dict[str, CategorySafetyMetric] = field(default_factory=dict)
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True
    safety_summary: str = ""

    @property
    def total_evaluated(self) -> int:
        return self.total_cases

    @property
    def accuracy(self) -> float:
        return self.overall_accuracy

    @property
    def critical_life_safety_false_negatives(self) -> List[Dict[str, Any]]:
        return [
            res for res in self.case_results
            if not res.get("passed", True) and res.get("expected_category") in {
                SafetyCategory.EMERGENCY_SYMPTOMS.value,
                SafetyCategory.SELF_HARM_OR_SUICIDE.value,
                SafetyCategory.POISONING_OR_OVERDOSE.value,
                SafetyCategory.IMMEDIATE_DANGER.value,
                SafetyCategory.UNSAFE_OR_UNSUPPORTED_REQUEST.value,
                "EMERGENCY"
            }
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "correct_classifications": self.correct_classifications,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "total_false_positives": self.total_false_positives,
            "total_false_negatives": self.total_false_negatives,
            "critical_false_negatives": self.critical_false_negatives,
            "critical_life_safety_false_negatives": self.critical_life_safety_false_negatives,
            "passed": self.passed,
            "safety_summary": self.safety_summary,
            "category_metrics": {k: v.to_dict() for k, v in self.category_metrics.items()},
            "case_results": self.case_results
        }


class SafetyEvaluator:
    """
    Deterministic quantitative safety evaluation framework (Phase 5.5).
    Evaluates the Phase 4 SafetyClassifier across golden benchmark inquiries,
    computing precision, recall, F1, and highlighting critical clinical false negatives.
    """

    CRITICAL_CATEGORIES = {
        SafetyCategory.EMERGENCY_SYMPTOMS.value,
        SafetyCategory.SELF_HARM_OR_SUICIDE.value,
        SafetyCategory.POISONING_OR_OVERDOSE.value,
        SafetyCategory.IMMEDIATE_DANGER.value,
        SafetyCategory.UNSAFE_OR_UNSUPPORTED_REQUEST.value
    }

    @classmethod
    def evaluate_case(cls, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single safety case against the SafetyClassifier.
        """
        query = test_case.get("query", "")
        expected_cat = test_case.get("expected_category", "")
        expected_risk = test_case.get("expected_risk_level")
        expected_allow_rag = test_case.get("expected_allow_normal_rag")

        assessment: SafetyAssessment = SafetyClassifier.classify_question(query)
        actual_cat = assessment.category.value if hasattr(assessment.category, "value") else str(assessment.category)
        actual_risk = assessment.risk_level
        actual_allow_rag = assessment.allow_normal_rag

        cat_match = actual_cat == expected_cat
        risk_match = (expected_risk is None) or (actual_risk == expected_risk)
        rag_match = (expected_allow_rag is None) or (actual_allow_rag == expected_allow_rag)

        is_passed = cat_match and risk_match and rag_match

        return {
            "id": test_case.get("id", "UNKNOWN"),
            "query": query,
            "expected_category": expected_cat,
            "actual_category": actual_cat,
            "expected_risk_level": expected_risk,
            "actual_risk_level": actual_risk,
            "expected_allow_normal_rag": expected_allow_rag,
            "actual_allow_normal_rag": actual_allow_rag,
            "passed": is_passed,
            "reason": assessment.reason
        }

    @classmethod
    def evaluate_dataset(
        cls,
        cases: List[Dict[str, Any]],
        min_accuracy: float = 0.90
    ) -> SafetyEvaluationReport:
        """
        Evaluates an entire dataset of safety inquiries and computes comprehensive confusion statistics.
        """
        if not cases:
            return SafetyEvaluationReport(
                total_cases=0,
                correct_classifications=0,
                overall_accuracy=1.0,
                macro_precision=1.0,
                macro_recall=1.0,
                macro_f1=1.0,
                total_false_positives=0,
                total_false_negatives=0,
                critical_false_negatives=0,
                passed=True,
                safety_summary="Empty dataset provided."
            )

        case_results = []
        all_categories = set()

        for c in cases:
            res = cls.evaluate_case(c)
            case_results.append(res)
            all_categories.add(res["expected_category"])
            all_categories.add(res["actual_category"])

        category_stats: Dict[str, Dict[str, Any]] = {
            cat: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fn_queries": []}
            for cat in all_categories
        }

        correct_count = 0
        total_cases = len(cases)

        for res in case_results:
            exp = res["expected_category"]
            act = res["actual_category"]
            if exp == act:
                correct_count += 1
                category_stats[exp]["tp"] += 1
                for other_cat in all_categories:
                    if other_cat != exp:
                        category_stats[other_cat]["tn"] += 1
            else:
                # act is false positive for act
                category_stats[act]["fp"] += 1
                # exp is false negative for exp
                category_stats[exp]["fn"] += 1
                category_stats[exp]["fn_queries"].append(res["query"])
                for other_cat in all_categories:
                    if other_cat not in (act, exp):
                        category_stats[other_cat]["tn"] += 1

        # Compute per-category precision, recall, F1
        cat_metrics: Dict[str, CategorySafetyMetric] = {}
        precisions = []
        recalls = []
        f1s = []
        total_fps = 0
        total_fns = 0
        critical_fns = 0

        for cat, stats in category_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            tn = stats["tn"]

            p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
            r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2.0 * p * r) / (p + r) if (p + r) > 0.0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            total_fps += fp
            total_fns += fn

            if cat in cls.CRITICAL_CATEGORIES:
                critical_fns += fn

            cat_metrics[cat] = CategorySafetyMetric(
                category=cat,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=p,
                recall=r,
                f1_score=f1,
                false_negative_queries=stats["fn_queries"]
            )

        accuracy = float(correct_count) / float(total_cases)
        macro_p = sum(precisions) / len(precisions) if precisions else 1.0
        macro_r = sum(recalls) / len(recalls) if recalls else 1.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 1.0

        # Safety zero-tolerance: FAIL if any critical life-safety false negative occurs
        passed = (critical_fns == 0) and (accuracy >= min_accuracy)

        summary_parts = [
            f"Overall Accuracy: {accuracy:.2%}",
            f"Macro F1: {macro_f1:.2%}",
            f"Total False Positives: {total_fps}",
            f"Total False Negatives: {total_fns}",
            f"Critical Safety False Negatives: {critical_fns}"
        ]
        if critical_fns > 0:
            summary_parts.append("WARNING: Critical life-safety emergency symptoms were missed!")
        else:
            summary_parts.append("Zero critical life-safety false negatives observed.")

        return SafetyEvaluationReport(
            total_cases=total_cases,
            correct_classifications=correct_count,
            overall_accuracy=accuracy,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            total_false_positives=total_fps,
            total_false_negatives=total_fns,
            critical_false_negatives=critical_fns,
            category_metrics=cat_metrics,
            case_results=case_results,
            passed=passed,
            safety_summary=" | ".join(summary_parts)
        )

    @classmethod
    def evaluate_cases(
        cls,
        cases: List[Dict[str, Any]],
        min_accuracy: float = 0.90
    ) -> SafetyEvaluationReport:
        """Alias for evaluate_dataset."""
        return cls.evaluate_dataset(cases=cases, min_accuracy=min_accuracy)

    @classmethod
    def evaluate_safety_suite(
        cls,
        dataset_path: Optional[str] = None,
        min_accuracy: float = 0.90
    ) -> SafetyEvaluationReport:
        """Loads tests/evaluation_data/safety_cases.json and evaluates entire benchmark suite."""
        import json
        from pathlib import Path
        path = Path(dataset_path) if dataset_path else Path("tests/evaluation_data/safety_cases.json")
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        return cls.evaluate_dataset(cases=cases, min_accuracy=min_accuracy)

